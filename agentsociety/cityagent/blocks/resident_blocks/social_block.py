# Due to the current limitations of the simulator's support, only NoneBlock, MessageBlock, and FindPersonBlock are available in the Dispatcher.

from typing import Any, Optional
import json_repair

from ....agent import (
    AgentToolbox,
    Block,
    FormatPrompt,
    BlockParams,
    DotDict,
    BlockContext,
)
from ....logger import get_logger
from ....memory import Memory
from ....agent.dispatcher import BlockDispatcher
from ..utils import TIME_ESTIMATE_PROMPT, clean_json_response
from ...sharing_params import SocietyAgentBlockOutput
from pydantic import Field


class MessagePromptManager:
    """
    Manages the creation of message prompts by dynamically formatting templates with agent-specific data.
    """

    def __init__(self):
        pass

    async def get_prompt(
        self, memory, step: dict[str, Any], environment_info: str, target: int, template: str
    ):
        """Generates a formatted prompt for message creation.

        Args:
            memory: Agent's memory to retrieve status data.
            step: Current workflow step containing intention and context.
            target: ID of the target agent for communication.
            template: Raw template string to be formatted.

        Returns:
            Formatted prompt string with placeholders replaced by agent-specific data.
        """
        # Retrieve data
        social_network = await memory.status.get("social_network") or []
        relationship = None
        for relation in social_network:
            if relation.target_id == target:
                relationship = relation
                break
        if relationship is None:
            get_logger().warning(f"MessagePromptManager: No relation found for target {target}")
            relationship_type = "unknown"
            relationship_strength = 0.0
        else:
            relationship_type = relationship.kind
            relationship_strength = relationship.strength
        chat_histories = await memory.status.get("chat_histories") or {}

        # Build discussion topic constraints
        discussion_constraint = ""
        topics = await memory.status.get("attitude")
        topics = topics.keys()
        if topics:
            topics = ", ".join(f'"{topic}"' for topic in topics)
            discussion_constraint = (
                f"Limit your discussion to the following topics: {topics}."
            )

        # Format prompt
        format_prompt = FormatPrompt(template)
        await format_prompt.format(
            name=await memory.status.get("name", "unknown"),
            gender=await memory.status.get("gender", "unknown"),
            occupation=await memory.status.get("occupation", "unknown"),
            education=await memory.status.get("education", "unknown"),
            personality=await memory.status.get("personality", "unknown"),
            emotion_types=await memory.status.get("emotion_types", "unknown"),
            thought=await memory.status.get("thought", "unknown"),
            background_story=await memory.status.get("background_story", "unknown"),
            relationship_type=relationship_type,
            relationship_strength=relationship_strength,
            chat_history=(
                chat_histories.get(target, "")
                if isinstance(chat_histories, dict)
                else ""
            ),
            intention=step.get("intention", ""),
            discussion_constraint=discussion_constraint,
            environment_info=environment_info,
        )

        return format_prompt.to_dialog()


class FindPersonBlock(Block):
    """
    Block for selecting an appropriate agent to socialize with based on relationship strength and context.
    """

    name = "FindPersonBlock"
    description = "Find a suitable person to socialize with"

    def __init__(self, toolbox: AgentToolbox, agent_memory: Memory):
        super().__init__(
            toolbox=toolbox,
            agent_memory=agent_memory,
        )

        self.prompt = """
Based on the following information, help me select the most suitable target to interact with:

1. Your Profile:
    - Gender: {gender}
    - Education: {education}
    - Personality: {personality}
    - Occupation: {occupation}
    - Background story: {background_story}

2. Your Current Intention: {intention}

3. Your Current Emotion: {emotion_types}

4. Your Current Thought: {thought}

5. Your social network (shown as id-to-relationship pairs):
    {friend_info}
    Note: For each target, the relationship strength (0-1) indicates how close we are

Please analyze and select:
1. The most appropriate target based on relationship strength and my current intention
2. Whether we should meet online or offline (online: chat, offline: meet in person)

Please output in JSON format, a dictionary:
{{
    "mode": "online" or "offline",
    "target_id": int
}}
        """

    async def forward(self, context: DotDict):
        
        # Get friends list and relationship strength
        my_social_network = await self.memory.status.get("social_network", [])
        if len(my_social_network) == 0:
            node_id = await self.memory.stream.add(
                topic="social",
                description="I can't find any target to contact with in my social network.",
            )
            return {
                "success": False,
                "evaluation": "No target found in social network.",
                "consumed_time": 0,
                "node_id": node_id,
            }

        # Different relationship types
        relationship_info = """
        My social network:
        """
        for relation in my_social_network:
            relationship_info += f"""
            - target_id: {relation.target_id}, relationship_type: {relation.kind}, relationship_strength: {relation.strength}
            """

        # Format the prompt
        formatted_prompt = FormatPrompt(self.prompt)
        await formatted_prompt.format(
            gender=str(await self.memory.status.get("gender")),
            education=str(await self.memory.status.get("education")),
            personality=str(await self.memory.status.get("personality")),
            occupation=str(await self.memory.status.get("occupation")),
            background_story=str(await self.memory.status.get("background_story")),
            intention=str(context["current_step"].get("intention", "socialize")),
            emotion_types=str(await self.memory.status.get("emotion_types")),
            thought=str(await self.memory.status.get("thought")),
            friend_info=relationship_info,
        )

        # Get LLM response
        response = await self.llm.atext_request(
            formatted_prompt.to_dialog(), timeout=300
        )

        try:
            # Parse the response
            response = json_repair.loads(response)
            mode = response["mode"] # type: ignore
            target_id = response["target_id"] # type: ignore

            # Validate the response format
            if not isinstance(mode, str):
                mode = str(mode)
            mode = mode.strip().lower()
            if mode not in ["online", "offline"]:
                raise ValueError("Invalid mode")
            if not isinstance(target_id, int):
                try:
                    target_id = int(str(target_id).strip())
                except Exception:
                    raise ValueError("Invalid target id")

            # Convert index to ID
            target = target_id
            if context is not None:
                context["target"] = target
        except Exception:
            # If parsing fails, select the friend with the strongest relationship as the default option
            get_logger().warning(f"Error parsing find person response: {response}")
            target = my_social_network[0].target_id
            mode = "online"
        
        # force to be online
        mode = "online"

        node_id = await self.memory.stream.add(
            topic="social",
            description=f"I selected the friend {target} for {mode} interaction",
        )
        return {
            "success": True,
            "evaluation": f"Selected friend {target} for {mode} interaction",
            "consumed_time": 0,
            "mode": mode,
            "target": target,
            "node_id": node_id,
        }


class MessageBlock(Block):
    """Generate and send messages"""

    name = "MessageBlock"
    description = "Send social message to someone (including online and offline, phone call, social post, etc.)"

    def __init__(self, toolbox: AgentToolbox, agent_memory: Memory):
        super().__init__(
            toolbox=toolbox,
            agent_memory=agent_memory,
        )
        self.find_person_block = FindPersonBlock(toolbox, agent_memory)

        # configurable fields
        self.default_message_template = """
My name is {name}, I am a {gender}
My occupation is {occupation}. 
My education level is {education}.
My personality is {personality}.
My current emotion is: {emotion_types}.
My current thought is: {thought}.
My background story is: {background_story}.

Now, I want to generate a social message to a target, my relationship with him/her:
Our relationship type is: {relationship_type}
Our relationship strength: {relationship_strength} (0-1, higher is stronger)
My previous chat history with him/her is:
{chat_history}

My intention is: {intention}.

Environment Information:
{environment_info}

Please generate a natural and contextually appropriate message.
Keep it under 100 characters.
The message should reflect my personality and background.

{discussion_constraint}

Please output the message from a first-person perspective, without any other text
"""

        self.prompt_manager = MessagePromptManager()

    async def forward(self, context: DotDict):

        result = await self.find_person_block.forward(context)
        if not result["success"]:
            return {
                "success": False,
                "evaluation": "Could not find target for message",
                "consumed_time": 0,
                "node_id": result["node_id"],
            }
        target = result["target"]

        environment_info = """"""
        weather = self.environment.sense("weather")
        if weather:
            environment_info += f"\nCurrent weather: {weather}"
        other_info = self.environment.sense("other_information")
        if other_info:
            environment_info += f"\nOther information: {other_info}"
        if not environment_info:
            environment_info = "No environment information"

        # Get formatted prompt using prompt manager
        formatted_prompt = await self.prompt_manager.get_prompt(
            self.memory,
            context["current_step"],
            environment_info,
            target,
            self.default_message_template,
        )

        # Generate message
        message = await self.llm.atext_request(formatted_prompt, timeout=300)

        # Update chat history with proper format
        chat_histories = await self.memory.status.get("chat_histories") or {}
        if not isinstance(chat_histories, dict):
            chat_histories = {}
        if target not in chat_histories:
            chat_histories[target] = ""
        elif len(chat_histories[target]) > 0:
            chat_histories[target] += ", "
        chat_histories[target] += f"me: {message}"

        await self.memory.status.update("chat_histories", chat_histories)

        # Send message
        await self.agent.send_message_to_agent(target, message, type="social")
        node_id = await self.memory.stream.add(
            topic="social", description=f"I sent a message to {target}: {message}"
        )
        return {
            "success": True,
            "evaluation": f"Sent message to {target}: {message}",
            "consumed_time": 10,
            "node_id": node_id,
        }


class SocialBlockParams(BlockParams): ...


class SocialBlockContext(BlockContext):
    target: Optional[int] = Field(
        default=None,
        description="The target agent id that the agent is going to socialize with",
    )


class SocialBlock(Block):
    """
    Orchestrates social interactions by dispatching to appropriate sub-blocks.
    """

    ParamsType = SocialBlockParams
    OutputType = SocietyAgentBlockOutput
    ContextType = SocialBlockContext
    NeedAgent = True
    name = "SocialBlock"
    description = "Do social interactions, for example, find a friend, send a message, and other social activities."
    actions = {
        "find_person": "Support the find person action, determine the social target.",
        "message": "Support the message action, send a message to the social target.",
        "social_none": "Support other social operations",
    }

    def __init__(
        self,
        toolbox: AgentToolbox,
        agent_memory: Memory,
        block_params: Optional[SocialBlockParams] = None,
    ):
        super().__init__(
            toolbox=toolbox,
            agent_memory=agent_memory,
            block_params=block_params,
        )
        self.message_block = MessageBlock(toolbox, agent_memory)
        self.trigger_time = 0
        self.token_consumption = 0

    async def forward(self, agent_context: DotDict) -> SocietyAgentBlockOutput:
        """Main entry point for social interactions. Dispatches to sub-blocks based on context.

        Args:
            step: Workflow step containing intention and metadata.
            context: Additional execution context.

        Returns:
            Result dict from the executed sub-block.
        """
        self.trigger_time += 1
        context = agent_context | self.context
        self.message_block.set_agent(self.agent)
        # Select the appropriate sub-block using dispatcher
        # Execute the selected sub-block and get the result
        result = await self.message_block.forward(context)
        return self.OutputType(**result)
