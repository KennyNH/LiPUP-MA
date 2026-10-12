import random
from typing import Any, Optional

import json_repair
from pydantic import Field

from ....agent import (
    AgentToolbox,
    Block,
    FormatPrompt,
    BlockParams,
    BlockContext,
    DotDict,
)
from ....logger import get_logger
from ....memory import Memory
from ....agent.dispatcher import BlockDispatcher
from ..utils import TIME_ESTIMATE_PROMPT, clean_json_response
from ...sharing_params import SocietyAgentBlockOutput


class OtherBlockParams(BlockParams):
    sleep_time_estimation_prompt: str = Field(
        default='',
        description="Used to determine the sleep time",
    )


class OtherBlockContext(BlockContext): ...


class OtherBlock(Block):
    """Orchestration block for managing specialized sub-blocks (SleepBlock/OtherNoneBlock).

    Attributes:
        sleep_block (SleepBlock): Specialized block for sleep actions.
        other_none_block (OtherNoneBlock): Fallback block for generic actions.
        trigger_time (int): Counter for block activation frequency.
        token_consumption (int): Accumulated LLM token usage.
        dispatcher (BlockDispatcher): Router for selecting appropriate sub-blocks.
    """

    ParamsType = OtherBlockParams
    OutputType = SocietyAgentBlockOutput
    ContextType = OtherBlockContext
    name = "OtherBlock"
    description = "Individual decisions or behaviors, including the following 6 subtypes: sleeping, eating, working, studying, entertainment, sporting, trivial."
    actions = {
        "sleep": "Support the sleep action",
        "other": "Support other actions",
    }

    def __init__(
        self,
        toolbox: AgentToolbox,
        agent_memory: Memory,
        block_params: Optional[OtherBlockParams] = None,
    ):
        super().__init__(
            toolbox=toolbox,
            agent_memory=agent_memory,
            block_params=block_params,
        )
        self.trigger_time = 0
        self.token_consumption = 0

    async def forward(self, agent_context: DotDict) -> SocietyAgentBlockOutput:
        """Route workflow steps to appropriate sub-blocks and track resource usage.

        Args:
            context: Workflow context containing plan and metadata.

        Returns:
            Execution result from the selected sub-block.
        """
        self.trigger_time += 1
        consumption_start = (
            self.llm.prompt_tokens_used + self.llm.completion_tokens_used
        )

        context = agent_context | self.context

        node_id = await self.memory.stream.add(
            # topic="other",
            topic=context.current_step_subtype,
            description=f"I {context['current_intention']}"
        )
        return self.OutputType(
            success=True,
            evaluation=f"Successfully {context['current_intention']}",
            consumed_time=int(context["current_step_expected_duration"]),
            node_id=node_id,
        )
