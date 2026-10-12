from typing import Optional

from pydantic import Field
from ..agent import (
    BlockOutput,
    AgentParams,
    AgentContext,
)
from ..agent.dispatcher import DISPATCHER_PROMPT
from .blocks.needs_block import INITIAL_NEEDS_PROMPT
from .blocks.plan_block import DETAILED_PLAN_PROMPT


class SocietyAgentConfig(AgentParams):
    """Configuration for society agent."""
    block_dispatch_prompt: str = Field(
        default=DISPATCHER_PROMPT,
        description="The prompt used for the block dispatcher, there is a variable 'intention' in the prompt, which is the intention of the task, used to select the most appropriate block",
    )

    enable_cognition: bool = Field(
        default=True, description="Whether to enable cognition"
    )
    
    enable_object_reflection: bool = Field(
        default=True, description="Whether to enable reflection over the object"
    )

    # Maxlow's Needs
    need_initialization_prompt: str = Field(
        default=INITIAL_NEEDS_PROMPT, description="Initial needs prompt"
    )

    # Planned-Behavior
    max_plan_steps: int = Field(
        default=5, description="Maximum number of steps in a plan"
    )
    plan_generation_prompt: str = Field(
        default=DETAILED_PLAN_PROMPT, description="Plan generation prompt"
    )


class SocietyAgentBlockOutput(BlockOutput):
    success: bool  # whether the action is executed successfully
    evaluation: str  # evaluation of the action
    consumed_time: int  # time consumed by the action
    node_id: Optional[int]  # stream memory node id
    
class SocietyAgentMobilityBlockOutput(BlockOutput):
    success: bool  # whether the action is executed successfully
    evaluation: str  # evaluation of the action
    consumed_time: int  # time consumed by the action
    node_id: Optional[int]  # stream memory node id
    to_place: Optional[int]


class SocietyAgentContext(AgentContext):
    # Normal Information
    current_time: str = Field(
        default="Don't know", description="Current time, in HH:MM:SS format"
    )
    current_need: str = Field(
        default="Don't know", description="Current need"
    )  # runtime
    current_intention: str = Field(
        default="Don't know", description="Current intention"
    )  # runtime
    current_emotion: str = Field(default="Don't know", description="Current emotion")
    current_thought: str = Field(default="Don't know", description="Current thought")
    current_location: str = Field(default="Don't know", description="Current location")
    
    current_step_type: str = Field(default="Don't know", description="Current step type")
    current_step_subtype: str = Field(default="Don't know", description="Current step subtype")
    current_step_expected_duration: str = Field(default="Don't know", description="Current_step_expected_duration")
    current_plan_related_places: list = Field(default=[], description="related places of each plan")

    # Environment Information
    area_information: str = Field(default="Don't know", description="Area information")
    weather: str = Field(default="Don't know", description="Weather information")
    temperature: str = Field(
        default="Don't know", description="Temperature information"
    )
    other_information: str = Field(
        default="Don't know", description="Other information"
    )

    # Plan Information
    plan_target: str = Field(default="Don't know", description="Plan target")
    max_plan_steps: int = Field(
        default=6, description="Maximum number of steps in a plan"
    )

    # Block Execution Information
    current_step: dict = Field(default={}, description="Current step")
    plan_context: dict = Field(default={}, description="Plan context")
    
    daily_plan_activation: bool = Field(default=False, description="Whether conduct daily plan?")
    daily_plan: list = Field(default=[], description="daily plan (activity sequence))")
    daily_plan_histories: list = Field(default=[], description="daily plan histories (history activity sequence))")
    
    move_status: bool = Field(default=False, description="Whether move")
    # current_aoi_info: str = Field(default="Don't know", description="cuirrent AOI Info")
