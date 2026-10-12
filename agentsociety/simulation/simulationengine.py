"""
A clear version of the simulation.
"""
import math
import copy
import asyncio
import inspect
import json
import pprint
import os
import traceback
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from multiprocessing import cpu_count
from typing import Any, Callable, Literal, Optional, Union, cast
import networkx as nx
import yaml
from fastembed import SparseTextEmbedding

from ..agent import (
    Agent,
    AgentToolbox,
    BankAgentBase,
    CitizenAgentBase,
    FirmAgentBase,
    GovernmentAgentBase,
    MemoryAttribute,
    NBSAgentBase,
    SupervisorBase,
)
from ..agent.distribution import Distribution, DistributionConfig, DistributionType
from ..agent.memory_config_generator import (
    MemoryConfig,
    MemoryConfigGenerator,
    default_memory_config_citizen,
    default_memory_config_supervisor,
)

from ..configs import (
    AgentConfig,
    AgentFilterConfig,
    Config,
    WorkflowType,
)
from ..environment import EnvironmentStarter
from ..llm import LLM
from ..logger import get_logger, set_logger_level
from ..memory import Memory
from ..message import Message, MessageInterceptor, MessageKind, Messager
from ..s3 import S3Config
from ..storage import DatabaseWriter
from ..storage.type import (
    StorageExpInfo,
    StorageGlobalPrompt,
    StoragePendingSurvey,
    StorageProfile,
    StorageStatus,
)
from ..survey.models import Survey
from .type import ExperimentStatus, Logs
from ..cityagent.resident_agent import ResidentAgent

__all__ = ["SimulationEngine"]

MIN_ID = 1
MAX_ID = 100000000


def _set_default_agent_config(self: Config):
    """
    Validates configuration options to ensure the user selects the correct combination.
    - **Description**:
        - If citizens contains at least one CITIZEN type agent, automatically fills
            empty institution agent lists with default configurations.
        - Sets default memory_config_func for citizen agents if not specified.

    - **Returns**:
        - `AgentsConfig`: The validated configuration instance.
    """
    # Set default memory config function for citizens
    for agent_config in self.agents.citizens:
        if agent_config.memory_config_func is None:
            agent_config.memory_config_func = default_memory_config_citizen

    if self.agents.supervisor is not None:
        if self.agents.supervisor.memory_config_func is None:
            self.agents.supervisor.memory_config_func = default_memory_config_supervisor

    return self


def _init_agent_class(agent_config: AgentConfig, s3config: S3Config):
    """
    Initialize the agent class.

    - **Args**:
        - `agent_config` (AgentConfig): The agent configuration.

    - **Returns**:
        - `agents`: A list of tuples, each containing an agent class, a memory config generator, and an index.
    """
    agent_class: type[Agent] = agent_config.agent_class
    n: int = agent_config.number
    # memory config function
    memory_config_func = cast(
        Callable[
            [dict[str, Distribution], Optional[list[MemoryAttribute]]],
            MemoryConfig,
        ],
        agent_config.memory_config_func,
    )
    generator = MemoryConfigGenerator(
        memory_config_func,
        agent_class.StatusAttributes,
        agent_config.number,
        agent_config.memory_from_file,
        (
            agent_config.memory_distributions
            if agent_config.memory_distributions is not None
            else {}
        ),
        s3config,
    )
    # lazy generate memory values
    # param config
    agent_params = agent_config.agent_params
    if agent_params is None:
        agent_params = agent_class.ParamsType()
    else:
        agent_params = agent_class.ParamsType.model_validate(agent_params)
    blocks = agent_config.blocks
    agents = [(agent_class, generator, i, agent_params, blocks) for i in range(n)]
    return agents, generator


def evaluate_filter(filter_str: str, profile: dict) -> bool:
    """
    Evaluate a filter string against a profile dictionary.

    - **Args**:
        - `filter_str` (str): The filter string to evaluate, e.g. "${profile.age} > 0"
        - `profile` (dict): The profile dictionary to evaluate against

    - **Returns**:
        - `bool`: True if the filter matches, False otherwise

    - **Note**:
        - Returns False if profile is empty
        - Returns False if any key in filter_str is not in profile
    """
    # if profile is empty, return False
    if not profile:
        return False

    # check if all keys in filter_str are in profile
    import re

    pattern = r"\${profile\.([^}]+)}"
    required_keys = set(re.findall(pattern, filter_str))

    # if any required key is not in profile, return False
    for key in required_keys:
        # Handle nested keys
        current = profile
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]

    # replace all ${profile.xxx} with actual values
    for key in required_keys:
        # Get the value by traversing the nested dictionary
        current = profile
        for part in key.split("."):
            current = current[part]
        filter_str = filter_str.replace(f"${{profile.{key}}}", repr(current))

    # use eval to execute the expression
    try:
        return eval(filter_str)
    except Exception:
        return False


class SimulationEngine:
    def __init__(
        self,
        config: Config,
        tenant_id: str = "",

    ) -> None:
        self.temp_current_plan=None
        self._config = _set_default_agent_config(config)
        self.tenant_id = tenant_id
        # ====================
        # Initialize the logger
        # ====================
        set_logger_level(self._config.logging_level.upper())

        self.exp_id = str(config.exp.id)
        get_logger().debug(
            f"Creating SimulationEngine with config: {self._config.model_dump()} as exp_id={self.exp_id}"
        )

        # typing definition
        self._llm: Optional[LLM] = None
        self._environment: Optional[EnvironmentStarter] = None
        self._message_interceptor: Optional[MessageInterceptor] = None
        self._database_writer: Optional[DatabaseWriter] = None
        self._embedding: Optional[SparseTextEmbedding] = None
        self._id2agent: dict[int, Agent] = {}
        yaml_config = yaml.dump(
            self._config.model_dump(
                exclude_defaults=True,
                exclude_none=True,
                exclude={
                    "llm": {
                        "__all__": {"api_key": True},
                    },
                    "env": {
                        "db": {"pg_dsn": True},
                        "s3": True,
                    },
                },
            ),
            allow_unicode=True,
        )
        self._exp_info: StorageExpInfo = StorageExpInfo(
            id=self.exp_id,
            tenant_id=self.tenant_id,
            name=self.name,
            num_day=0,
            status=0,
            cur_day=0,
            cur_t=0.0,
            config=yaml_config,
            error="",
            input_tokens=0,
            output_tokens=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self._total_steps: int = 0
        self._messager: Optional[Messager] = None

        # simulation context - for information dump
        self.context = {}

        # filter base
        self._filter_base = {}



        self.memory_graph=nx.MultiGraph()



    async def _init_embedding(self):
        """Initialize embedding model with timeout."""
        try:
            # Create a task for embedding initialization
            init_task = asyncio.create_task(self._init_embedding_task())

            # Wait for the task with timeout
            try:
                await asyncio.wait_for(init_task, timeout=120)  # 2 minutes timeout
            except asyncio.TimeoutError:
                get_logger().error(
                    "Embedding model initialization timed out after 2 minutes. "
                    "Please check your HuggingFace connection and try again."
                )
                raise

        except Exception as e:
            get_logger().error(f"Failed to initialize embedding model: {str(e)}")
            raise

    async def _init_embedding_task(self):
        """Actual embedding initialization task."""
        self._embedding = SparseTextEmbedding(
            "Qdrant/bm25",
            cache_dir=os.path.join(self._config.env.home_dir, "huggingface_cache"),
            threads=cpu_count(),
        )
        get_logger().info("Embedding model initialized successfully")

    async def init(self):
        """Initialize all the components"""
        # ====================
        # Initialize the pgsql writer
        # ====================
        if self._config.env.db.enabled:
            get_logger().info("Initializing database writer...")
            self._database_writer = DatabaseWriter(
                self.tenant_id,
                self.exp_id,
                self._config.env.db,
                self._config.env.home_dir,
            )
            await self._database_writer.init()  # type: ignore
            get_logger().info("Database writer initialized")
            # save to local
            await self._database_writer.update_exp_info(self._exp_info)

        try:
            # ====================
            # Initialize the LLM
            # ====================
            get_logger().info("Initializing LLM...")
            self._llm = LLM(self._config.llm)
            get_logger().info("LLM initialized")

            # ====================
            # Initialize the environment
            # ====================
            get_logger().info("Initializing environment...")
            self._environment = EnvironmentStarter(
                self._config.map,
                self._config.exp.environment,
                self._config.env.s3,
                os.path.join(
                    self._config.env.home_dir,
                    "exps",
                    self.tenant_id,
                    self.exp_id,
                    "simulator_log",
                ),
                self._config.env.home_dir,
            )
            await self._environment.init()
            get_logger().info("Environment initialized")

            # ====================
            # Initialize the messager
            # ====================
            get_logger().info("Initializing messager...")
            if self._config.agents.supervisor is not None:
                self._message_interceptor = MessageInterceptor(
                    self._config.llm,
                )
            self._messager = Messager(exp_id=self.exp_id)
            get_logger().info("Messager initialized")

            # ====================
            # Initialize the embedding
            # ====================
            get_logger().info("Initializing embedding...")
            await self._init_embedding()
            assert self._embedding is not None, "Embedding is not initialized"
            get_logger().info("Embedding initialized")

            # ======================================
            # Initialize agents
            # ======================================
            agents = []  # (id, agent_class, generator, memory_index)
            next_id = 1
            defined_ids = set()  # used to check if the id is already defined

            def _find_next_id():
                nonlocal next_id  # Declare that we want to modify the outer variable
                while next_id in defined_ids:
                    next_id += 1
                if next_id > MAX_ID:
                    raise ValueError(
                        f"Agent ID {next_id} is greater than MAX_ID {MAX_ID}"
                    )
                defined_ids.add(next_id)
                return next_id

            citizen_ids = set()
            bank_ids = set()
            nbs_ids = set()
            government_ids = set()
            firm_ids = set()
            supervisor_ids = set()
            aoi_ids = self._environment.get_aoi_ids()

            aoi_id_home = []
            home_weights = []
            aoi_id_work = []
            work_weights = []
            work_type_weights = {
                "OF": 1.0,
                "SC": 0.5,
                "BS": 0.8,
                "ID": 0.5,
                "HS": 0.5,
                "RC": 0.2,
                "PK": 0.1,
            }
            for i in aoi_ids:
                aoi_info = self.environment.map.aois[i]
                land_use = aoi_info["urban_land_use"]
                if land_use == "RS":
                    area = float(aoi_info.get("area", 1.0))
                    if area <= 0:
                        area = 1.0
                    aoi_id_home.append(i)
                    home_weights.append(area)
                elif land_use in work_type_weights:
                    aoi_id_work.append(i)
                    work_weights.append(work_type_weights[land_use])
            if len(aoi_id_home) == 0:
                raise ValueError("No RS AOI found for home assignment.")
            if len(aoi_id_work) == 0:
                raise ValueError(
                    "No valid work AOI found. Expected one of OF/SC/BS/ID/HS/RC/PK."
                )
            # Check if any agent config uses memory_from_file
            agent_configs_normal = {
                "firms": [],
                "banks": [],
                "nbs": [],
                "governments": [],
                "citizens": [],
                "supervisor": [],
            }
            agent_configs_from_file = {
                "firms": [],
                "banks": [],
                "nbs": [],
                "governments": [],
                "citizens": [],
                "supervisor": [],
            }
            for agent_config in self._config.agents.firms:
                if agent_config.memory_from_file is None:
                    agent_configs_normal["firms"].append(agent_config)
                else:
                    agent_configs_from_file["firms"].append(agent_config)
            for agent_config in self._config.agents.banks:
                if agent_config.memory_from_file is None:
                    agent_configs_normal["banks"].append(agent_config)
                else:
                    agent_configs_from_file["banks"].append(agent_config)
            for agent_config in self._config.agents.nbs:
                if agent_config.memory_from_file is None:
                    agent_configs_normal["nbs"].append(agent_config)
                else:
                    agent_configs_from_file["nbs"].append(agent_config)
            for agent_config in self._config.agents.governments:
                if agent_config.memory_from_file is None:
                    agent_configs_normal["governments"].append(agent_config)
                else:
                    agent_configs_from_file["governments"].append(agent_config)
            for agent_config in self._config.agents.citizens:
                if agent_config.memory_from_file is None:
                    agent_configs_normal["citizens"].append(agent_config)
                else:
                    agent_configs_from_file["citizens"].append(agent_config)
            if self._config.agents.supervisor is not None:
                agent_config = self._config.agents.supervisor
                if agent_config.memory_from_file is None:
                    agent_configs_normal["supervisor"] = [agent_config]
                else:
                    agent_configs_from_file["supervisor"] = [agent_config]

            citizen_generators = []
            # Step 1: Process all agents with memory_from_file
            # Firms
            for agent_config in agent_configs_from_file["firms"]:
                agent_config = cast(AgentConfig, agent_config)
                agent_class = agent_config.agent_class
                agent_params = agent_config.agent_params
                if agent_params is None:
                    agent_params = agent_class.ParamsType()
                else:
                    agent_params = agent_class.ParamsType.model_validate(agent_params)
                blocks = agent_config.blocks
                # Create generator
                generator = MemoryConfigGenerator(
                    agent_config.memory_config_func,
                    agent_config.agent_class.StatusAttributes,
                    agent_config.number,
                    agent_config.memory_from_file,
                    (
                        agent_config.memory_distributions
                        if agent_config.memory_distributions is not None
                        else {}
                    ),
                    self._config.env.s3,
                )
                # Get agent data from file
                agent_data = generator.get_agent_data_from_file()
                # Extract IDs from agent data
                for agent_datum in agent_data:
                    agent_id = agent_datum.get("id")
                    assert (
                        agent_id is not None
                    ), "id is required in memory_from_file[Firms]"
                    assert (
                        agent_id >= MIN_ID
                    ), f"id {agent_id} is less than MIN_ID {MIN_ID}"
                    assert (
                        agent_id <= MAX_ID
                    ), f"id {agent_id} is greater than MAX_ID {MAX_ID}"
                    assert (
                        agent_id not in defined_ids
                    ), f"id {agent_id} is already defined"
                    defined_ids.add(agent_id)
                    firm_ids.add(agent_id)
                    agents.append(
                        (
                            agent_id,
                            agent_class,
                            generator,
                            agent_data.index(agent_datum),
                            agent_params,
                            blocks,
                        )
                    )

            # Banks
            for agent_config in agent_configs_from_file["banks"]:
                agent_config = cast(AgentConfig, agent_config)
                agent_class = agent_config.agent_class
                agent_params = agent_config.agent_params
                if agent_params is None:
                    agent_params = agent_class.ParamsType()
                else:
                    agent_params = agent_class.ParamsType.model_validate(agent_params)
                blocks = agent_config.blocks
                generator = MemoryConfigGenerator(
                    agent_config.memory_config_func,
                    agent_config.agent_class.StatusAttributes,
                    agent_config.number,
                    agent_config.memory_from_file,
                    (
                        agent_config.memory_distributions
                        if agent_config.memory_distributions is not None
                        else {}
                    ),
                    self._config.env.s3,
                )
                agent_data = generator.get_agent_data_from_file()
                for agent_datum in agent_data:
                    agent_id = agent_datum.get("id")
                    assert (
                        agent_id is not None
                    ), "id is required in memory_from_file[Banks]"
                    assert (
                        agent_id >= MIN_ID
                    ), f"id {agent_id} is less than MIN_ID {MIN_ID}"
                    assert (
                        agent_id <= MAX_ID
                    ), f"id {agent_id} is greater than MAX_ID {MAX_ID}"
                    assert (
                        agent_id not in defined_ids
                    ), f"id {agent_id} is already defined"
                    defined_ids.add(agent_id)
                    bank_ids.add(agent_id)
                    agents.append(
                        (
                            agent_id,
                            agent_class,
                            generator,
                            agent_data.index(agent_datum),
                            agent_params,
                            blocks,
                        )
                    )

            # NBS
            for agent_config in agent_configs_from_file["nbs"]:
                agent_config = cast(AgentConfig, agent_config)
                agent_class = agent_config.agent_class
                agent_params = agent_config.agent_params
                if agent_params is None:
                    agent_params = agent_class.ParamsType()
                else:
                    agent_params = agent_class.ParamsType.model_validate(agent_params)
                blocks = agent_config.blocks
                generator = MemoryConfigGenerator(
                    agent_config.memory_config_func,
                    agent_config.agent_class.StatusAttributes,
                    agent_config.number,
                    agent_config.memory_from_file,
                    (
                        agent_config.memory_distributions
                        if agent_config.memory_distributions is not None
                        else {}
                    ),
                    self._config.env.s3,
                )
                agent_data = generator.get_agent_data_from_file()
                for agent_datum in agent_data:
                    agent_id = agent_datum.get("id")
                    assert (
                        agent_id is not None
                    ), "id is required in memory_from_file[NBS]"
                    assert (
                        agent_id >= MIN_ID
                    ), f"id {agent_id} is less than MIN_ID {MIN_ID}"
                    assert (
                        agent_id <= MAX_ID
                    ), f"id {agent_id} is greater than MAX_ID {MAX_ID}"
                    assert (
                        agent_id not in defined_ids
                    ), f"id {agent_id} is already defined"
                    defined_ids.add(agent_id)
                    nbs_ids.add(agent_id)
                    agents.append(
                        (
                            agent_id,
                            agent_class,
                            generator,
                            agent_data.index(agent_datum),
                            agent_params,
                            blocks,
                        )
                    )

            # Governments
            for agent_config in agent_configs_from_file["governments"]:
                agent_config = cast(AgentConfig, agent_config)
                agent_class = agent_config.agent_class
                agent_params = agent_config.agent_params
                if agent_params is None:
                    agent_params = agent_class.ParamsType()
                else:
                    agent_params = agent_class.ParamsType.model_validate(agent_params)
                blocks = agent_config.blocks
                generator = MemoryConfigGenerator(
                    agent_config.memory_config_func,
                    agent_config.agent_class.StatusAttributes,
                    agent_config.number,
                    agent_config.memory_from_file,
                    (
                        agent_config.memory_distributions
                        if agent_config.memory_distributions is not None
                        else {}
                    ),
                    self._config.env.s3,
                )
                agent_data = generator.get_agent_data_from_file()
                for agent_datum in agent_data:
                    agent_id = agent_datum.get("id")
                    assert (
                        agent_id is not None
                    ), "id is required in memory_from_file[Governments]"
                    assert (
                        agent_id >= MIN_ID
                    ), f"id {agent_id} is less than MIN_ID {MIN_ID}"
                    assert (
                        agent_id <= MAX_ID
                    ), f"id {agent_id} is greater than MAX_ID {MAX_ID}"
                    assert (
                        agent_id not in defined_ids
                    ), f"id {agent_id} is already defined"
                    defined_ids.add(agent_id)
                    government_ids.add(agent_id)
                    agents.append(
                        (
                            agent_id,
                            agent_class,
                            generator,
                            agent_data.index(agent_datum),
                            agent_params,
                            blocks,
                        )
                    )

            # Citizens
            for agent_config in agent_configs_from_file["citizens"]:
                agent_config = cast(AgentConfig, agent_config)
                agent_class = agent_config.agent_class
                agent_params = agent_config.agent_params
                if agent_params is None:
                    agent_params = agent_class.ParamsType()
                else:
                    agent_params = agent_class.ParamsType.model_validate(agent_params)
                blocks = agent_config.blocks
                generator = MemoryConfigGenerator(
                    agent_config.memory_config_func,
                    agent_config.agent_class.StatusAttributes,
                    agent_config.number,
                    agent_config.memory_from_file,
                    (
                        agent_config.memory_distributions
                        if agent_config.memory_distributions is not None
                        else {}
                    ),
                    self._config.env.s3,
                )
                citizen_generators.append(generator)
                agent_data = generator.get_agent_data_from_file()
                for agent_datum in agent_data:
                    agent_id = agent_datum.get("id")
                    assert (
                        agent_id is not None
                    ), "id is required in memory_from_file[Citizens]"
                    assert (
                        agent_id >= MIN_ID
                    ), f"id {agent_id} is less than MIN_ID {MIN_ID}"
                    assert (
                        agent_id <= MAX_ID
                    ), f"id {agent_id} is greater than MAX_ID {MAX_ID}"
                    assert (
                        agent_id not in defined_ids
                    ), f"id {agent_id} is already defined"
                    defined_ids.add(agent_id)
                    citizen_ids.add(agent_id)
                    agents.append(
                        (
                            agent_id,
                            agent_class,
                            generator,
                            agent_data.index(agent_datum),
                            agent_params,
                            blocks,
                        )
                    )

            # Supervisor
            assert (
                len(agent_configs_from_file["supervisor"]) <= 1
            ), "only one or zero supervisor is allowed"
            supervisor: Optional[SupervisorBase] = None
            for agent_config in agent_configs_from_file["supervisor"]:
                agent_config = cast(AgentConfig, agent_config)
                generator = MemoryConfigGenerator(
                    agent_config.memory_config_func,
                    agent_config.agent_class.StatusAttributes,
                    agent_config.number,
                    agent_config.memory_from_file,
                    (
                        agent_config.memory_distributions
                        if agent_config.memory_distributions is not None
                        else {}
                    ),
                    self._config.env.s3,
                )
                agent_data = generator.get_agent_data_from_file()
                for agent_datum in agent_data:
                    agent_id = agent_datum.get("id")
                    assert (
                        agent_id is not None
                    ), "id is required in memory_from_file[Supervisor]"
                    assert (
                        agent_id >= MIN_ID
                    ), f"id {agent_id} is less than MIN_ID {MIN_ID}"
                    assert (
                        agent_id <= MAX_ID
                    ), f"id {agent_id} is greater than MAX_ID {MAX_ID}"
                    assert (
                        agent_id not in defined_ids
                    ), f"id {agent_id} is already defined"
                    defined_ids.add(agent_id)
                    supervisor_ids.add(agent_id)
                    memory_config = generator.generate(i=0)
                    memory_init = Memory(
                        environment=self.environment,
                        embedding=self._embedding,
                        memory_config=memory_config,
                    )
                    # build blocks
                    if agent_config.blocks is not None:
                        blocks = [
                            block_type(
                                llm=self._llm,
                                environment=self.environment,
                                agent_memory=memory_init,
                                block_params=block_params,
                            )
                            for block_type, block_params in agent_config.blocks.items()
                        ]
                    else:
                        blocks = None
                    # build agent
                    if agent_config.agent_params is None:
                        agent_params = agent_config.agent_class.ParamsType()
                    else:
                        agent_params = agent_config.agent_class.ParamsType.model_validate(
                            agent_config.agent_params
                        )
                    supervisor = agent_config.agent_class(
                        id=agent_id,
                        name=f"{agent_config.agent_class.__name__}_{agent_id}",
                        toolbox=AgentToolbox(
                            llm=self._llm,
                            environment=self.environment,
                            messager=self.messager,
                            embedding=self._embedding,
                            database_writer=self._database_writer,
                        ),
                        memory=memory_init,
                        agent_params=agent_params,
                        blocks=blocks,
                    )
                    # set supervisor
                    assert (
                        self._message_interceptor is not None
                    ), "message interceptor is not set"
                    await self._message_interceptor.set_supervisor(supervisor)
                    break

            get_logger().info(
                f"{len(defined_ids)} defined ids found in memory_config_files"
            )

            # Step 2: Process all agents without memory_from_file
            for agent_config in agent_configs_normal["firms"]:
                agent_config = cast(AgentConfig, agent_config)
                # Handle distribution-based configuration as before
                if agent_config.memory_distributions is None:
                    agent_config.memory_distributions = {}
                assert (
                    "aoi_id" not in agent_config.memory_distributions
                ), "aoi_id is not allowed to be set in memory_distributions because it will be generated in the initialization"
                agent_config.memory_distributions["aoi_id"] = DistributionConfig(
                    dist_type=DistributionType.CHOICE,
                    choices=list(aoi_ids),
                )
                firm_classes, _ = _init_agent_class(agent_config, self._config.env.s3)
                firms = [
                    (_find_next_id(), *firm_class)
                    for i, firm_class in enumerate(firm_classes)
                ]
                firm_ids.update([firm[0] for firm in firms])
                agents += firms

            for agent_config in agent_configs_normal["banks"]:
                bank_classes, _ = _init_agent_class(agent_config, self._config.env.s3)
                banks = [
                    (_find_next_id(), *bank_class)
                    for i, bank_class in enumerate(bank_classes)
                ]
                bank_ids.update([bank[0] for bank in banks])
                agents += banks

            for agent_config in agent_configs_normal["nbs"]:
                nbs_classes, _ = _init_agent_class(agent_config, self._config.env.s3)
                nbs = [
                    (_find_next_id(), *nbs_class)
                    for i, nbs_class in enumerate(nbs_classes)
                ]
                nbs_ids.update([nbs[0] for nbs in nbs])
                agents += nbs

            for agent_config in agent_configs_normal["governments"]:
                government_classes, _ = _init_agent_class(
                    agent_config, self._config.env.s3
                )
                governments = [
                    (_find_next_id(), *government_class)
                    for i, government_class in enumerate(government_classes)
                ]
                government_ids.update([government[0] for government in governments])
                agents += governments

            for agent_config in agent_configs_normal["citizens"]:
                citizen_classes, generator = _init_agent_class(
                    agent_config, self._config.env.s3
                )
                citizen_generators.append(generator)
                citizens = [
                    (_find_next_id(), *citizen_class)
                    for i, citizen_class in enumerate(citizen_classes)
                ]
                citizen_ids.update([citizen[0] for citizen in citizens])
                agents += citizens

            for agent_config in agent_configs_normal["supervisor"]:
                supervisor_classes, _ = _init_agent_class(
                    agent_config, self._config.env.s3
                )
                supervisors = [
                    (_find_next_id(), *supervisor_class)
                    for i, supervisor_class in enumerate(supervisor_classes)
                ]
                supervisor_ids.update([supervisor[0] for supervisor in supervisors])
            
            # Step 3: Insert essential distributions for citizens
            memory_distributions = {}
            memory_distributions["home_aoi_id"] = DistributionConfig(
                dist_type=DistributionType.CHOICE,
                choices=list(aoi_id_home),
                weights=home_weights,
            )
            memory_distributions["work_aoi_id"] = DistributionConfig(
                dist_type=DistributionType.CHOICE,
                choices=list(aoi_id_work),
                weights=work_weights,
            )
            for generator in citizen_generators:
                generator.merge_distributions(memory_distributions)

            get_logger().info(
                f"agents: len(citizens)={len(citizen_ids)}, len(firms)={len(firm_ids)}, len(banks)={len(bank_ids)}, len(nbs)={len(nbs_ids)}, len(governments)={len(government_ids)}"
            )
            self._environment.economy_client.set_ids(
                citizen_ids=citizen_ids,
                firm_ids=firm_ids,
                bank_ids=bank_ids,
                nbs_ids=nbs_ids,
                government_ids=government_ids,
            )

            # ====================================
            # Initialize the agent objects
            # ====================================
            agent_toolbox = AgentToolbox(
                llm=self.llm,
                environment=self.environment,
                messager=self.messager,
                embedding=self._embedding,
                database_writer=self._database_writer,
            )
            get_logger().info("Initializing the agents...")
            to_return = {}
            for agent_init in agents:
                (
                    id,
                    agent_class,
                    memory_config_generator,
                    index_for_generator,
                    agent_params,
                    blocks,
                ) = agent_init
                memory_config = memory_config_generator.generate(index_for_generator)
                to_return[id] = (agent_class, deepcopy(memory_config))
                # Initialize Memory with the unified config             
                memory_init = Memory(
                    environment=self.environment,
                    embedding=self._embedding,
                    memory_config=memory_config,
                )
                # # build blocks
                if blocks is not None:
                    blocks = [
                        block_type(
                            toolbox=agent_toolbox,
                            agent_memory=memory_init,
                            block_params=block_type.ParamsType.model_validate(
                                block_params
                            ),
                        )
                        for block_type, block_params in blocks.items()
                    ]
                else:
                    blocks = None
                # build agent
                agent = agent_class(
                    id=id,
                    name=f"{agent_class.__name__}_{id}",
                    toolbox=agent_toolbox,
                    memory=memory_init,
                    agent_params=agent_params,
                    blocks=blocks,
                )
                self._id2agent[id] = agent
            get_logger().info("-----Initializing by running agent.init() ...")
            tasks = []
            channels = []
            for agent in self._id2agent.values():
                tasks.append(agent.init())
                channels.append(f"exps:{self.exp_id}:agents:{agent.id}:*")
            await asyncio.gather(*tasks)
            get_logger().info("-----Initializing by exporting profiles ...")
            profiles = []
            for agent in self._id2agent.values():
                profile = await agent.status.export(
                    [
                        "name",
                        "gender",
                        "age",
                        "education",
                        "occupation",
                        "marriage_status",
                        "persona",
                        "background_story",
                    ]
                )
                profile["id"] = agent.id
                profiles.append(
                    StorageProfile(
                        id=agent.id,
                        name=profile.get("name", ""),
                        profile=json.dumps(
                            {
                                k: v
                                for k, v in profile.items()
                                if k not in {"id", "name", "social_network"}
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
            if self._database_writer is not None:
                await self._database_writer.write_profiles(profiles)  # type:ignore
            get_logger().info("-----Initializing embeddings ...")
            embedding_tasks = []
            for agent in self._id2agent.values():
                embedding_tasks.append(agent.memory.initialize_embeddings())
            await asyncio.gather(*embedding_tasks)

            get_logger().info("Agents initialized")

            for agent_id, (agent_class, memory_config) in to_return.items():
                self._filter_base[agent_id] = (agent_class, memory_config)

            get_logger().info("Agents initialized")
            # step 1 tick to make the initialization complete
            await self.environment.step(1)
            get_logger().info("run 1 tick to make the initialization complete")

            # ===================================
            # save the experiment info
            # ===================================
            await self._save_exp_info()
            self._save_context()
            get_logger().info("Experiment info saved")

            # ===================================
            # run init functions
            # ===================================
            init_funcs = self._config.agents.init_funcs
            for init_func in init_funcs:
                if inspect.iscoroutinefunction(init_func):
                    await init_func(self)
                else:
                    init_func(self)

            '''
            
            从这里开始初始化graph
            '''
            aois = self.environment.map.aois  # 一个字典，key是aoi id,后面是这个aoi的属性加字典,可能考虑简化:
            residents = self._id2agent

            for node_id, attributes in aois.items():
                attributes_selected = {'id':attributes["id"],"positions":attributes["positions"],
                                       "area":attributes["area"],"name":attributes["name"],"urban_land_use":attributes["urban_land_use"]}
                self.memory_graph.add_node(node_id, **attributes_selected,node_type="aoi_node")

            for resident_id, agentinfo in residents.items(): #把home work属性添加上去

                if type(agentinfo) == ResidentAgent:
                    home_id = agentinfo.status._data["home"]["aoi_position"]["aoi_id"]
                    workplace_id = agentinfo.status._data["work"]["aoi_position"]["aoi_id"]
                    self.memory_graph.add_node(resident_id, home_aoi_id=home_id, work_aoi_id=workplace_id,node_type="resident_node")

            #print(list(self.memory_graph.nodes(data=True)))


        except Exception as e:
            get_logger().error(f"Init error: {str(e)}\n{traceback.format_exc()}")
            self._exp_info.status = ExperimentStatus.ERROR.value
            self._exp_info.error = str(e)
            await self._save_exp_info()

            raise e
        get_logger().info("Init functions run")
        get_logger().info("Simulation initialized")

    async def close(self):
        """Close all the components"""

        # ===================================
        # close groups
        # ===================================

        get_logger().info("Closing agent groups...")
        close_tasks = []
        for agent in self._id2agent.values():
            close_tasks.append(agent.close())  # type:ignore
        await asyncio.gather(*close_tasks)
        get_logger().info("Agents closed")

        if self._environment is not None:
            get_logger().info("Closing environment...")
            await self._environment.close()
            self._environment = None
            get_logger().info("Environment closed")

    @property
    def name(self):
        return self._config.exp.name

    @property
    def config(self):
        return self._config

    @property
    def llm(self):
        assert self._llm is not None, "llm is not initialized"
        return self._llm

    @property
    def enable_database(self):
        return self._config.env.db.enabled

    @property
    def database_writer(self):
        assert self._database_writer is not None, "database writer is not initialized"
        return self._database_writer

    @property
    def environment(self):
        assert self._environment is not None, "environment is not initialized"
        return self._environment

    @property
    def messager(self):
        assert self._messager is not None, "messager is not initialized"
        return self._messager

    async def _extract_target_agent_ids(
        self, target_agent: Optional[Union[list[int], AgentFilterConfig]] = None
    ) -> list[int]:
        if target_agent is None:
            raise ValueError("target_agent is required")
        elif isinstance(target_agent, list):
            return target_agent
        elif isinstance(target_agent, AgentFilterConfig):
            return await self.filter(
                types=target_agent.agent_class, filter_str=target_agent.filter_str # type: ignore
            )
        else:
            raise ValueError("target_agent must be a list of int or AgentFilterConfig")

    async def gather(
        self,
        content: str,
        target_agent_ids: Optional[list[int]] = None,
        flatten: bool = False,
        keep_id: bool = False,
    ) -> Union[dict[int, Any], list[Any]]:
        """
        Collect specific information from agents.

        - **Description**:
            - Asynchronously gathers specified content from targeted agents within all groups.

        - **Args**:
            - `content` (str): The information to collect from the agents.
            - `target_agent_ids` (Optional[List[int]], optional): A list of agent IDs to target. Defaults to None, meaning all agents are targeted.
            - `flatten` (bool, optional): Whether to flatten the result. Defaults to False.
            - `keep_id` (bool, optional): Whether to keep the agent IDs in the result. Defaults to False.

        - **Returns**:
            - Result of the gathering process as returned by each group's `gather` method.
        """
        results = {}
        if target_agent_ids is None:
            target_agent_ids = list(self._id2agent.keys())
        if content == "stream_memory":
            for agent in self._id2agent.values():
                if agent.id in target_agent_ids:
                    results[agent.id] = await agent.stream.get_all()
        else:
            for agent in self._id2agent.values():
                if agent.id in target_agent_ids:
                    results[agent.id] = await agent.status.get(content)
        if flatten:
            if not keep_id:
                data_flatten = []
                for _, data in results.items():
                    data_flatten.append(data)
                return data_flatten
            else:
                data_flatten = {}
                for id, data in results.items():
                    data_flatten[id] = data
                return data_flatten
        else:
            return results

    async def filter(
        self,
        types: Optional[tuple[type[Agent]]] = None,
        filter_str: Optional[str] = None,
    ) -> list[int]:
        """
        Filter out agents of specified types or with matching key-value pairs.

        - **Args**:
            - `types` (Optional[Tuple[Type[Agent]]], optional): Types of agents to filter for. Defaults to None.
            - `filter_str` (Optional[str], optional): Filter string to match in agent attributes. Defaults to None.

        - **Raises**:
            - `ValueError`: If neither types nor filter_str are provided.

        - **Returns**:
            - `List[int]`: A list of filtered agent UUIDs.
        """
        if not types and not filter_str:
            return list(self._id2agent.keys())
        # filter by types first
        if types:
            filtered_ids = [
                agent_id
                for agent_id, (agent_class, _) in self._filter_base.items()
                if any(issubclass(agent_class, t) for t in types)
            ]
        else:
            filtered_ids = list(self._id2agent.keys())

        # filter by filter_str
        if filter_str:
            filtered_ids = [
                agent_id
                for agent_id in filtered_ids
                if evaluate_filter(filter_str, self._filter_base[agent_id][1])
            ]

        return filtered_ids

    async def update_environment(self, key: str, value: str):
        """
        Update the environment variables for the simulation and all agent groups.

        - **Args**:
            - `key` (str): The environment variable key to update.
            - `value` (str): The new value for the environment variable.
        """
        self.environment.update_environment(key, value)

    async def update(
        self,
        target_agent_ids: list[int],
        target_key: str,
        content: Any,
        query: bool = False,
    ):
        """
        Update the memory of specified agents.

        - **Args**:
            - `target_agent_id` (list[int]): The IDs of the target agents to update.
            - `target_key` (str): The key in the agent's memory to update.
            - `content` (Any): The new content to set for the target key.
        """
        get_logger().debug(f"-----Updating {target_key} for agent {target_agent_ids}")
        tasks = []
        for agent_id in target_agent_ids:
            agent = self._id2agent[agent_id]
            if query:
                agent.gather_results[target_key] = content
            tasks.append(agent.status.update(target_key, content))
        await asyncio.gather(*tasks)

    async def economy_update(
        self,
        target_agent_id: int,
        target_key: str,
        content: Any,
        mode: Literal["replace", "merge"] = "replace",
    ):
        """
        Update economic data for a specified agent.

        - **Args**:
            - `target_agent_id` (int): The ID of the target agent whose economic data to update.
            - `target_key` (str): The key in the agent's economic data to update.
            - `content` (Any): The new content to set for the target key.
            - `mode` (Literal["replace", "merge"], optional): Mode of updating the economic data. Defaults to "replace".
        """
        await self.environment.economy_client.update(
            id=target_agent_id, key=target_key, value=content, mode=mode
        )

    async def send_survey(
        self,
        survey: Survey,
        agent_ids: list[int] = [],
        survey_day: Optional[int] = None,
        survey_t: Optional[float] = None,
        is_pending_survey: bool = False,
        pending_survey_id: Optional[int] = None,
    ) -> dict[int, str]:
        """
        Send a survey to specified agents.

        - **Args**:
            - `survey` (Survey): The survey object to send.
            - `agent_ids` (List[int], optional): List of agent IDs to receive the survey. Defaults to an empty list.
            - `survey_day` (int, optional): The day of the survey. Defaults to None.
            - `survey_t` (float, optional): The time of the survey. Defaults to None.
            - `is_pending_survey` (bool, optional): Whether the survey is a pending survey. Defaults to False.
            - `pending_survey_id` (int, optional): The ID of the pending survey. Defaults to None.

        - **Returns**:
            - `dict[int, str]`: A dictionary mapping agent IDs to their survey responses.
        """
        survey_tasks = []
        for agent_id in agent_ids:
            agent = self._id2agent[agent_id]
            if isinstance(agent, CitizenAgentBase):
                survey_tasks.append(
                    agent._handle_survey_with_storage(
                        survey,
                        survey_day,
                        survey_t,
                        is_pending_survey,
                        pending_survey_id,
                    )
                )
            else:
                get_logger().error(
                    f"Agent {agent_id} is not a citizen agent, so skip the survey"
                )
        survey_responses = await asyncio.gather(*survey_tasks)
        return {
            agent_id: response
            for agent_id, response in zip(agent_ids, survey_responses)
        }

    async def send_interview_message(
        self,
        question: str,
        agent_ids: list[int],
    ):
        """
        Send an interview message to specified agents.

        - **Args**:
            - `question` (str): The content of the message to send.
            - `agent_ids` (list[int]): A list of IDs for the agents to receive the message.

        - **Returns**:
            - None
        """
        day, t = self.environment.get_datetime()
        interview_tasks = []
        for agent_id in agent_ids:
            agent = self._id2agent[agent_id]
            if isinstance(agent, CitizenAgentBase):
                interview_tasks.append(
                    agent._handle_interview_with_storage(
                        Message(
                            from_id=None,
                            to_id=agent_id,
                            payload={"content": question},
                            kind=MessageKind.USER_CHAT,
                            day=day,
                            t=t,
                        )
                    )
                )
            else:
                get_logger().error(
                    f"Agent {agent_id} is not a citizen agent, so skip the interview"
                )
        interview_responses = await asyncio.gather(*interview_tasks)
        return {
            agent_id: response
            for agent_id, response in zip(agent_ids, interview_responses)
        }

    async def send_intervention_message(
        self, intervention_message: str, agent_ids: list[int]
    ):
        """
        Send an intervention message to specified agents.

        - **Description**:
            - Send an intervention message to specified agents.

        - **Args**:
            - `intervention_message` (str): The content of the intervention message to send.
            - `agent_ids` (list[int]): A list of agent IDs to receive the intervention message.
        """
        react_tasks = []
        for agent_id in agent_ids:
            agent = self._id2agent[agent_id]
            if isinstance(agent, CitizenAgentBase):
                react_tasks.append(agent.react_to_intervention(intervention_message))
            else:
                get_logger().error(
                    f"Agent {agent_id} is not in the group, so skip the intervention"
                )
        await asyncio.gather(*react_tasks)

    async def _save_exp_info(self) -> None:
        """Async save experiment info to YAML file and pgsql"""
        self._exp_info.updated_at = datetime.now(timezone.utc)
        if self.enable_database:
            assert self._database_writer is not None
            await self._database_writer.update_exp_info(self._exp_info)  # type: ignore

    async def _save_global_prompt(self, prompt: str, day: int, t: float):
        """Save global prompt"""
        prompt_info = StorageGlobalPrompt(
            day=day,
            t=t,
            prompt=prompt,
            created_at=datetime.now(timezone.utc),
        )
        if self.enable_database:
            assert self._database_writer is not None
            await self._database_writer.write_global_prompt(prompt_info)  # type:ignore

    async def _gather_and_update_context(
        self, target_agent_ids: list[int], key: str, save_as: str
    ):
        """Gather and update the context"""
        try:
            values = await self.gather(
                key, target_agent_ids, flatten=True, keep_id=True
            )
            self.context[save_as] = values
        except Exception as e:
            get_logger().error(
                f"Error saving context: {str(e)}\n{traceback.format_exc()}"
            )
            self.context[save_as] = {}

    def _save_context(self):
        fs_client = self._config.env.fs_client
        json_bytes = json.dumps(self.context, indent=2, ensure_ascii=False).encode(
            "utf-8"
        )
        fs_client.upload(
            data=json_bytes,
            remote_path=f"exps/{self.tenant_id}/{self.exp_id}/artifacts.json",
        )

    # ====================
    # Message Handling Methods
    # ====================
    async def _message_dispatch(self):
        """
        Dispatches messages received via Message to the appropriate agents.
        """
        # Step 1: Fetch messages
        messages = await self.messager.fetch_received_messages()
        get_logger().info(f"Received {len(messages)} messages")

        try:
            # Step 2: Distribute messages to corresponding Agents
            # Separate messages into agent messages and aoi messages
            agent_messages = defaultdict(list)  # Dict[agent_id, list[Message]]
            aoi_messages = []  # List[Message]

            for message in messages:
                if message.kind in [MessageKind.AGENT_CHAT, MessageKind.USER_CHAT]:
                    agent_id = message.to_id
                    if agent_id in self._id2agent:
                        agent_messages[agent_id].append(message)
                elif message.kind in [
                    MessageKind.AOI_MESSAGE_REGISTER,
                    MessageKind.AOI_MESSAGE_CANCEL,
                ]:
                    aoi_messages.append(message)

            # Process agent messages in parallel for different agents
            async def process_agent_messages(agent_id: int, messages: list[Message]):
                agent = self._id2agent[agent_id]
                if isinstance(agent, CitizenAgentBase):
                    for message in messages:
                        if message.kind == MessageKind.AGENT_CHAT:
                            await agent._handle_agent_chat_with_storage(message)
                        elif message.kind == MessageKind.USER_CHAT:
                            await agent._handle_interview_with_storage(message)
                else:
                    get_logger().error(
                        f"Agent {agent_id} is not a citizen agent, so skip the message dispatch"
                    )

            # Process agent messages in parallel
            agent_tasks = [
                process_agent_messages(agent_id, msgs)
                for agent_id, msgs in agent_messages.items()
            ]
            await asyncio.gather(*agent_tasks)

            # Process aoi messages
            for message in aoi_messages:
                agent_id = message.from_id
                if message.kind == MessageKind.AOI_MESSAGE_REGISTER:
                    self.environment.register_aoi_message(
                        agent_id, message.to_id, message.payload["content"]
                    )
                elif message.kind == MessageKind.AOI_MESSAGE_CANCEL:
                    self.environment.cancel_aoi_message(agent_id, message.to_id)
        except Exception as e:
            get_logger().error(f"Error dispatching message: {e}")
            import traceback

            get_logger().error(f"Error dispatching message: {traceback.format_exc()}")

    async def _save(self, day: int, t: int):
        """
        Saves the current status of the agents at a given point in the simulation.

        - **Args**:
            - `day` (int): The day number in the simulation time.
            - `t` (int): The tick or time unit in the simulation day.
        """
        if self._database_writer is None:
            return
        created_at = datetime.now(timezone.utc)
        # =========================
        # build statuses data
        # =========================
        statuses = []
        for agent in self._id2agent.values():
            if isinstance(agent, CitizenAgentBase):
                position = await agent.status.get("position")
                x = position["xy_position"]["x"]
                y = position["xy_position"]["y"]
                lng, lat = self.environment.projector(x, y, inverse=True)
                if "aoi_position" in position:
                    parent_id = position["aoi_position"]["aoi_id"]
                elif "lane_position" in position:
                    parent_id = position["lane_position"]["lane_id"]
                else:
                    parent_id = None
                current_plan = await agent.status.get("current_plan", {})
                if current_plan is not None and current_plan:
                    step_index = current_plan.get("index", 0)
                    steps = current_plan.get("steps", [])
                    if not isinstance(steps, list) or len(steps) == 0:
                        action = "Planning"
                    else:
                        if not isinstance(step_index, int):
                            try:
                                step_index = int(step_index)
                            except Exception:
                                step_index = 0
                        if step_index < 0:
                            step_index = 0
                        if step_index >= len(steps):
                            # Keep save path robust even if current_plan index drifts.
                            step_index = len(steps) - 1
                        step_item = steps[step_index] if isinstance(steps[step_index], dict) else {}
                        action = step_item.get("intention", "Planning")
                else:
                    action = "Planning"
                status_summary = await agent.status.get("status_summary", "Nothing")
                status = StorageStatus(
                    id=agent.id,
                    day=day,
                    t=t,
                    lng=lng,
                    lat=lat,
                    parent_id=parent_id,
                    action=action,
                    status=status_summary,
                    created_at=created_at,
                )
                statuses.append(status)
            elif isinstance(
                agent, (FirmAgentBase, BankAgentBase, NBSAgentBase, GovernmentAgentBase)
            ):
                status_summary = await agent.status.get("status_summary", "Nothing")
                status = StorageStatus(
                    id=agent.id,
                    day=day,
                    t=t,
                    lng=None,
                    lat=None,
                    parent_id=None,
                    action="",
                    status=status_summary,
                    created_at=created_at,
                )
                statuses.append(status)
            else:
                raise ValueError(f"Unknown agent type: {type(agent)}")
        if self._database_writer is not None:
            await self._database_writer.write_statuses(  # type:ignore
                statuses
            )

    async def delete_agents(self, target_agent_ids: list[int]):
        """
        Delete the specified agents.

        - **Args**:
            - `target_agent_ids` (list[int]): The IDs of the agents to delete.
        """
        tasks = []
        for agent_id in target_agent_ids:
            agent = self._id2agent[agent_id]
            tasks.append(agent.close())
        await asyncio.gather(*tasks)
        for agent_id in target_agent_ids:
            del self._id2agent[agent_id]

    async def next_round(self):
        """
        Proceed to the next round of the simulation.
        """
        get_logger().info("Start entering the next round of the simulation")
        tasks = []
        for agent in self._id2agent.values():
            tasks.append(agent.reset())  # type:ignore
        await asyncio.gather(*tasks)
        await self.environment.step(1)
        get_logger().info("Finished entering the next round of the simulation")

    async def step(self, num_environment_ticks: int = 1) -> Logs:
        """
        Execute one step of the simulation where each agent performs its forward action.

        - **Description**:
            - Checks if new agents need to be inserted based on the current day of the simulation. If so, it inserts them.
            - Executes the forward method for each agent group to advance the simulation by one step.
            - Saves the state of all agent groups after the step has been completed.
            - Optionally extracts metrics if the current step matches the interval specified for any metric extractors.

        - **Args**:
            - `num_environment_ticks` (int): The number of ticks for the environment to step forward.

        - **Raises**:
            - `RuntimeError`: If there is an error during the execution of the step, it logs the error and rethrows it as a RuntimeError.

        - **Returns**:
            - `Logs`: The logs of the simulation.
        """
        try:
            # ======================
            # run a step
            # ======================
            day, t = self.environment.get_datetime()
            get_logger().info(
                f"Start simulation day {day} at {t}, {math.floor(t/3600)} h {(t%3600-1)/60} m, step {self._total_steps}"
            )
            await self._message_dispatch()
            # main agent workflow
            agents = list(self._id2agent.values())
            tasks = [agent.run() for agent in agents]   #就是这一步，会产生所有agent block furwardy以及反思等
            first_pass = await asyncio.gather(*tasks, return_exceptions=True)
            agent_time_log = []
            for agent, result in zip(agents, first_pass):
                if isinstance(result, Exception):
                    get_logger().warning(
                        f"Agent {getattr(agent, 'id', 'unknown')} run failed in this step: "
                        f"{type(result).__name__}: {result}. Retrying up to 10 times."
                    )
                    retry_ok = False
                    retry_result = -1.0
                    for retry_idx in range(10):
                        try:
                            retry_result = await agent.run()
                            retry_ok = True
                            break
                        except Exception as e2:
                            get_logger().warning(
                                f"Agent {getattr(agent, 'id', 'unknown')} retry "
                                f"{retry_idx + 1}/10 failed: {type(e2).__name__}: {e2}"
                            )
                    if not retry_ok:
                        get_logger().error(
                            f"Agent {getattr(agent, 'id', 'unknown')} all retries failed in this step."
                        )
                    # Keep simulation running; mark this agent as failed in this step if all retries fail.
                    agent_time_log.append(retry_result)
                else:
                    agent_time_log.append(result)
            simulator_log = (
                self.environment.get_log_list()
                + self.environment.economy_client.get_log_list()
            )
            log = Logs(
                llm_log=self.llm.get_log_list(),
                simulator_log=simulator_log,
                agent_time_log=agent_time_log,
            )
            self.llm.clear_log_list()
            self.environment.clear_log_list()
            self.environment.economy_client.clear_log_list()

            # gather query
            gather_queries = {}
            for agent in self._id2agent.values():
                if agent.gather_query:
                    gather_queries[agent.id] = agent.gather_query

            get_logger().debug(f"({day}-{t}) Finished agent forward steps")
            # ======================
            # log the simulation results
            # ======================
            all_logs = Logs(
                llm_log=[],
                simulator_log=[],
                agent_time_log=[],
            )
            all_logs.append(log)
            # ======================
            # save the experiment info
            # ======================
            self._exp_info.status = ExperimentStatus.RUNNING.value
            self._exp_info.cur_day = day
            self._exp_info.cur_t = t
            for log in all_logs.llm_log:
                self._exp_info.input_tokens += log.get("input_tokens", 0)
                self._exp_info.output_tokens += log.get("output_tokens", 0)
            await self._save_exp_info()
            self._save_context()
            # ======================
            # process gather queries
            # ======================
            for agent_id, group_queries in gather_queries.items():
                for query_key, query in group_queries.items():
                    result = await self.gather(query.key, query.target_agent_ids, flatten=query.flatten, keep_id=query.keep_id)  # type: ignore
                    await self.update([agent_id], query.key, result, query=True)  # type: ignore

            # ======================
            # save the simulation results
            # ======================
            await self._save(day, t)
            # save global prompt
            await self._save_global_prompt(
                prompt=self.environment.get_environment(),
                day=day,
                t=t,
            )
            get_logger().debug(f"({day}-{t}) Finished saving simulation results")
            # ======================
            # forward message
            # ======================
            all_messages = await self.messager.fetch_pending_messages()
            get_logger().info(
                f"({day}-{t}) Finished fetching pending messages. {len(all_messages)} messages fetched."
            )

            if self._message_interceptor is not None:
                all_messages = await self._message_interceptor.forward(all_messages)
            # ======================
            # fetch pending dialogs from USER
            # ======================
            if self.enable_database:
                pending_dialogs = await self._database_writer.fetch_pending_dialogs()  # type: ignore
                get_logger().info(
                    f"({day}-{t}) Finished fetching pending dialogs. {len(pending_dialogs)} dialogs fetched."
                )
                user_messages = []
                for pending_dialog in pending_dialogs:
                    user_messages.append(
                        Message(
                            from_id=None,
                            to_id=pending_dialog.agent_id,
                            payload={"content": pending_dialog.content},
                            created_at=pending_dialog.created_at,
                            kind=MessageKind.USER_CHAT,
                            day=pending_dialog.day,
                            t=pending_dialog.t,
                            extra={"pending_dialog_id": pending_dialog.id},
                        )
                    )
                all_messages += user_messages
            # dispatch messages to each agent group based on their to_id
            await self.messager.set_received_messages(all_messages)
            get_logger().info(f"({day}-{t}) Finished setting received messages")
            # ======================
            # handle pending surveys
            # ======================
            if self.enable_database:
                pending_surveys = await self._database_writer.fetch_pending_surveys()  # type: ignore
                get_logger().info(
                    f"({day}-{t}) Finished fetching pending surveys. {len(pending_surveys)} surveys fetched."
                )
                pending_surveys = cast(list[StoragePendingSurvey], pending_surveys)
                for pending_survey in pending_surveys:
                    try:
                        pending_survey.data["id"] = pending_survey.survey_id
                        survey = Survey.model_validate(pending_survey.data)
                    except Exception as e:
                        get_logger().error(
                            f"Error validating survey data: {str(e)}\n{traceback.format_exc()}"
                        )
                        continue
                    await self.send_survey(
                        survey,
                        [pending_survey.agent_id],
                        pending_survey.day,
                        pending_survey.t,
                        is_pending_survey=True,
                        pending_survey_id=pending_survey.id,
                    )
            # ======================
            # Log metrics from environment
            # ======================
            metrics = await self.environment.get_metrics()
            if self.enable_database:
                await self._database_writer.log_metric(metrics)
            get_logger().debug(f"({day}-{t}) Finished simulator sync")
            # ======================
            # go to next step
            # ======================
            self._total_steps += 1


            '''
            补充的接口
            1.
            '''


            agent_dict = self._id2agent
            for id, agent in agent_dict.items():
                # print(id)
                if type(agent) == ResidentAgent:  #id >= 1: #对于每一个city agent
                    cur_plan = await agent.memory.status.get("current_plan", False)
                    if isinstance(cur_plan, dict):
                        current_step_info = cur_plan.get("steps", [])
                    else:
                        current_step_info = []
                    # print("\n")
                    # print("*"*60)
                    # print("id: ",id)
                    # print("\nthe overall plan:", await agent.memory.status.get("current_plan", False))
                    # print(  "\ncurrent step info", current_step_info, self.environment.get_datetime())
                    reflection_info = agent.memory._objmem.obj_memories
                    # print("\nall reflection", reflection_info)
                    # print("\ntemp_obj_memory", agent.memory.objmem.temp_obj_memory)   #表示已反思的内容
                    # print("\nmove buffer", agent.move_buffer)  #表示已完成的步骤
                    # print("*"*60)
                    # print("\n")

                    '''
                    overall plans是规划一次，可能会包含很多step去按顺序执行，只是执行了，evaluation会变成true，
                    current step info有每一步信息，其中当前步执行的是蓝色的字；
                    duration是plan时的时间，evaluation time是实际执行时间，可能不一样（比如模拟单步太长）
                    
                    每一个规划步骤，会先执行check and update step 首先会结束上一步的动作，更新时间之类的信息；
                    其次开始下一步动作，更新evaluation信息,或者直接产生memory并规划新动作；
                    
                    所以如果一个动作结束完打印plan，是没有更新最后一步的动作信息的；
                    temp里面保留的也是上一个时间步结束完的信息，所以需要在check and update step的时候就把信息保存在buffer里面
                    
                    
                    '''
                    #self.memory_graph.add_edge()
                    '''
                    在第一步时先添加和初始位置的关系；
                    如果产生了移动：判断是不是第一次产生（不重复，即唯一一次移动行为） 添加relation node并连接居民
                    如果有新的想法：连接id和想法
                    '''

                    '''
                    第一部分，先添加主观反思信息
                    '''
                    obj_reflection=copy.deepcopy(agent.memory.objmem.temp_obj_memory) #
                    if obj_reflection: #说明已经完成了一个plan
                        for aoi_id, comments in obj_reflection.items(): #目前只添加了评论，没添加路径
                            self.memory_graph.add_edge(id,int(aoi_id), edge_type="obj_reflection",comments=list(comments.values()),day=day,t=t)  #这个day t信息按照绝对时间，和模拟起始时间无关
                        agent.memory.objmem.temp_obj_memory=None

                        '''
                         第二部分，添加客观移动信息
                         '''


                    if agent.move_buffer:
                        step=agent.move_buffer
                        mobility_info = [step["position"], step["evaluation"]["to_place"], step["start_time"],
                                         step["evaluation"]["consumed_time"]]  # 起点，终点，起始时间(从模拟是开始计算），耗时(min)
                        node_name = str([mobility_info[0], mobility_info[1]])
                        if not self.memory_graph.has_node(node_name):
                            self.memory_graph.add_node(node_name, node_type="relation_node")
                            self.memory_graph.add_edges_from([(node_name, mobility_info[0]),
                                                              (node_name, mobility_info[1])],
                                                             edge_type="aoi_relation")

                        self.memory_graph.add_edge(
                            id,
                            node_name,
                            edge_type="mobility",
                            start_timestamp=step["start_time"],
                            mobility_minutes=step["evaluation"]["consumed_time"],
                            intention=step.get("intention"),
                            distance_m=step.get("evaluation", {}).get("distance_m"),
                        )

                        agent.move_buffer=None



            await self.environment.step(num_environment_ticks)

            return all_logs
        except Exception as e:
            get_logger().error(f"Simulation error: {str(e)}\n{traceback.format_exc()}")
            raise RuntimeError(str(e)) from e

    def _reset_transient_step_state(self):
        """
        Best-effort cleanup of per-step transient states before retrying a failed step.
        This avoids repeating crashes caused by stale temporary data.
        """
        for agent in self._id2agent.values():
            try:
                if hasattr(agent, "move_buffer"):
                    agent.move_buffer = None
                mem = getattr(agent, "memory", None)
                objmem = getattr(mem, "objmem", None)
                if objmem is not None and hasattr(objmem, "temp_obj_memory"):
                    objmem.temp_obj_memory = None
            except Exception as e:
                get_logger().warning(
                    f"Transient state reset failed for agent {getattr(agent, 'id', 'unknown')}: {e}"
                )

    async def _step_with_retry(self, num_environment_ticks: int = 1, max_retries: int = 10) -> Logs:
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                return await self.step(num_environment_ticks)
            except Exception as e:
                last_err = e
                get_logger().warning(
                    f"Step failed (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{type(e).__name__}: {e}"
                )
                self._reset_transient_step_state()
        raise RuntimeError(str(last_err)) from last_err

    async def run_one_day(
        self,
        ticks_per_step: int,
    ):
        """
        Run the simulation for a day.

        - **Args**:
            - `ticks_per_step` (int): The number of ticks per step.

        - **Description**:
            - Updates the experiment status to running and sets up monitoring for the experiment's status.
            - Runs the simulation loop until the end time, which is calculated based on the current time and the number of days to simulate.
            - After completing the simulation, updates the experiment status to finished, or to failed if an exception occurs.

        - **Raises**:
            - `RuntimeError`: If there is an error during the simulation, it logs the error and updates the experiment status to failed before rethrowing the exception.

        - **Returns**:
            - None
        """
        logs = Logs(
            llm_log=[],
            simulator_log=[],
            agent_time_log=[],
        )
        start_day, _ = self.environment.get_datetime()
        num_steps = 0
        while True:
            this_logs = await self._step_with_retry(ticks_per_step, max_retries=10)



            logs.append(this_logs)
            day, _ = self.environment.get_datetime()
            num_steps += 1
            if day != start_day:
                #exit(-1)
                break
        return logs

    async def run(self):
        """
        Run the simulation following the workflow in the config.
        """
        logs = Logs(
            llm_log=[],
            simulator_log=[],
            agent_time_log=[],
        )
        try:
            for step in self.config.exp.workflow:

                # results = {}
                # if target_agent_ids is None:
                #     target_agent_ids = list(self._id2agent.keys())
                # if content == "stream_memory":
                #     for agent in self._id2agent.values():
                #         if agent.id in target_agent_ids:
                #             results[agent.id] = await agent.stream.get_all()
                # else:
                #     for agent in self._id2agent.values():
                #         if agent.id in target_agent_ids:
                #             results[agent.id] = await agent.status.get(content)




                get_logger().info(
                    f"Running workflow: type: {step.type} - description: {step.description}"
                )
                if step.type == WorkflowType.STEP:
                    for _ in range(step.steps):
                        log = await self._step_with_retry(step.ticks_per_step, max_retries=10)
                        logs.append(log)
                elif step.type == WorkflowType.RUN:
                    days = int(step.days)
                    remain = step.days - days
                    for _ in range(days):
                        day_ok = False
                        last_day_err = None
                        for day_retry_idx in range(10):
                            try:
                                log = await self.run_one_day(step.ticks_per_step)
                                day_ok = True
                                break
                            except Exception as e:
                                last_day_err = e
                                get_logger().warning(
                                    f"run_one_day failed (retry {day_retry_idx + 1}/10): "
                                    f"{type(e).__name__}: {e}"
                                )
                                self._reset_transient_step_state()
                        if not day_ok:
                            raise RuntimeError(str(last_day_err)) from last_day_err
                        logs.append(log)
                    if remain > 0.001:
                        ticks_remain = int(remain * 24 * 60 * 60 / step.ticks_per_step)
                        for _ in range(ticks_remain):
                            log = await self._step_with_retry(step.ticks_per_step, max_retries=10)
                            logs.append(log)
                elif step.type == WorkflowType.INTERVIEW:
                    target_agents = step.target_agent
                    interview_message = step.interview_message
                    assert interview_message is not None
                    assert target_agents is not None
                    target_agent_ids = await self._extract_target_agent_ids(
                        target_agents
                    )
                    await self.send_interview_message(
                        interview_message, target_agent_ids
                    )
                elif step.type == WorkflowType.SURVEY:
                    assert step.target_agent is not None
                    assert step.survey is not None
                    target_agent_ids = await self._extract_target_agent_ids(
                        step.target_agent
                    )
                    await self.send_survey(step.survey, target_agent_ids)
                elif step.type == WorkflowType.ENVIRONMENT_INTERVENE:
                    assert step.key is not None
                    assert step.value is not None
                    await self.update_environment(step.key, step.value)
                elif step.type == WorkflowType.UPDATE_STATE_INTERVENE:
                    assert step.key is not None
                    assert step.value is not None
                    assert step.target_agent is not None
                    target_agent_ids = await self._extract_target_agent_ids(
                        step.target_agent
                    )
                    await self.update(target_agent_ids, step.key, step.value)
                elif step.type == WorkflowType.MESSAGE_INTERVENE:
                    assert step.intervene_message is not None
                    assert step.target_agent is not None
                    target_agent_ids = await self._extract_target_agent_ids(
                        step.target_agent
                    )
                    await self.send_intervention_message(
                        step.intervene_message, target_agent_ids
                    )
                elif step.type == WorkflowType.NEXT_ROUND:
                    await self.next_round()
                elif step.type == WorkflowType.DELETE_AGENT:
                    assert step.target_agent is not None
                    target_agent_ids = await self._extract_target_agent_ids(
                        step.target_agent
                    )
                    await self.delete_agents(target_agent_ids)
                elif step.type == WorkflowType.SAVE_CONTEXT:
                    assert step.target_agent is not None
                    assert step.key is not None
                    assert step.save_as is not None
                    target_agent_ids = await self._extract_target_agent_ids(
                        step.target_agent
                    )
                    await self._gather_and_update_context(
                        target_agent_ids, step.key, step.save_as
                    )
                elif step.type == WorkflowType.INTERVENE:
                    get_logger().warning(
                        "MESSAGE_INTERVENE is not fully implemented yet, it can only influence the congnition of target agents"
                    )
                    assert step.target_agent is not None
                    assert step.intervene_message is not None
                    target_agent_ids = await self._extract_target_agent_ids(
                        step.target_agent
                    )
                    await self.send_intervention_message(
                        step.intervene_message, target_agent_ids
                    )
                elif step.type == WorkflowType.FUNCTION:
                    assert step.func is not None
                    assert not isinstance(step.func, str)
                    await step.func(self)
                else:
                    raise ValueError(f"Unknown workflow type: {step.type}")
                self._save_context()

                '''
                下面这块是新增的用于获取接口
                '''


        except Exception as e:
            get_logger().error(f"Simulation error: {str(e)}\n{traceback.format_exc()}")
            self._exp_info.status = ExperimentStatus.ERROR.value
            self._exp_info.error = str(e)
            self._save_context()
            await self._save_exp_info()

            raise RuntimeError(str(e)) from e
        self._exp_info.status = ExperimentStatus.FINISHED.value
        self._save_context()
        await self._save_exp_info()
        return logs, self.memory_graph
