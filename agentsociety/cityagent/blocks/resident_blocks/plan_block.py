import re
from typing import Any, Optional, Tuple


import json_repair

from ....agent import AgentToolbox, Agent, Block, FormatPrompt, DotDict
from ....logger import get_logger
from ....memory import Memory
from ..utils import clean_json_response

DETAILED_PLAN_PROMPT = """As an intelligent agent's decision system, please help me determine a suitable activity which conforms to you overall plan while satisfying my current need.
The Environment will influence the choice of steps.

Current weather: {weather}
Current temperature: {temperature}
Other information: 
-------------------------
{other_info}
-------------------------

Current location: {current_location}
Current time: {current_time}
My income/consumption level: {consumption_level}
My occupation: {occupation}
My age: {age}
My emotion: {emotion_types}
My thought: {thought}

(1) GENERATE "activity":
Today's overall plan: {daily_plan}
Today's history activities: {daily_plan_histories}
**Please prioritize the history activities as outlined in the plan. Be aware of my current time and location. The generated activity must strictly adhere to the provided daily plan. While creative rephrasing is permitted, the output must faithfully reflect the original schedule without substantial deviation.**
However, if I have urgent and infrequent needs, address those first. You may refer to and adjust the activities listed below.
Current need: Need to satisfy {current_need}, reference activities: {options}. Check whether the needs are really urgent, and avoid impacting the overall plan severely. 
Eating or sleeping are really important, do not skip them. Social is also necessary if I have spare time. DO NOT SKIP SLEEP!
Please also be mindful of my schedule. If I am busy, kindly adjust the plan accordingly.

(2) GENERATE "evaluation":
Please evaluate and determine the most appropriate activity based on these three dimensions:
1. Attitude: Personal preference and evaluation of the activity
2. Subjective Norm: Social environment and others' views on this behavior
3. Perceived Control: Difficulty and controllability of executing this activity

(3) GENERATE "activity_steps":
Also, please help me generate specific execution steps based on the planned activity, each step contains "type", "intention" and "duration". Notes are below:
1. **Type can only be one of these three: mobility, social, other
    1.1 mobility: Decisions or behaviors related to large-scale spatial movement, such as location selection, finding or going to a place, etc. Subtype is "mobility".
    1.2 social: Decisions or behaviors related to social interaction, such as finding contacts, chatting with friends, etc. Subtype is "social".
    1.3 other: Other types of individual decisions or behaviors, including the following 6 subtypes: sleeping, eating, working, studying, entertainment, sporting, trivial.**
2. Steps should only include steps necessary to fulfill the target (limited to {max_plan_steps} steps)
3. Intention in each step should be concise and clear
4. Duration in each step should be reasonable integar (measured in minutes), be aware of the overall plan and my profile information. The "mobility" or "social" type doesn't require exact duration, fill in with "TBD". 
5. If the step is "mobility" type, either find a new place or return known places (e.g., home, workplace). Only moving between places (e.g., POI, AOI) requires "mobility" step. 
6. **If the activity does not require moving, there's no need to generate multiple steps, just generate one step same with the activity. You should be aware of my current time and location.**
7. In the "activity_steps", except for "trivial" subtype, the types ("mobility", "social", "sleeping", "eating", "working", "studying", "entertainment", "sporting") can only appear for at most once.
8. If the activity can be performed in the current location, DO NOT generate "mobility" behaviors! No need to move in this case.

**Note: Apart from being at home and workplace, please generate as many daily activity types, as your life is not just work and rest.
So please try your best go outside and move to different places, to have more activities to meet your various needs, based on the provided information.
Make sure your plan is reasonable and the mobility step should be macro and involve sufficient distance (e.g. from one place (area of interest) to another)
not micro and within the same place (e.g., within single residential area).**
 

Please response in json format (Do not return any other text), example (**'duration' is str not int!**):
{{
    "activity": "have lunch at home",
    "evaluation": {{
        "attitude": "Attitude score for the activity (0-1)",
        "subjective_norm": "Subjective norm score (0-1)", 
        "perceived_control": "Perceived control score (0-1)",
        "reasoning": "Specific reasons for selecting this activity"
    }}
    "activity_steps": [
        {{
            "intention": "return home",
            "type": "mobility",
            "subtype": "mobility",
            "duration": "TBD"
        }},
        {{
            "intention": "cook food",
            "type": "other",
            "subtype": "trivial",
            "duration": "20"
        }},
        {{
            "intention": "have meal",
            "type": "other",
            "subtype": "eating",
            "duration": "25"
        }}
    ]
}}
All output text values must be in English.
"""

DAILY_PLAN_PROMPT = """As an intelligent agent's decision system, please help me generate a plan for today.
The Environment will influence the choice of steps.

Current weather: ${context.weather}
Current temperature: ${context.temperature}
Other information: 
-------------------------
${context.other_information}
-------------------------

Current time: ${context.current_time}
My income/consumption level: ${profile.consumption}
My occupation: ${profile.occupation}
My age: ${profile.age}
My emotion: ${profile.emotion_types}
My thought: ${context.current_thought}

Notes:
1. The current time and all other timestamps should be provided in 24-hour format. The plan must starts from the current time.
2. The daily plan should consist of a sequence of planned activities along with their expected time ranges. Activity types and names can be creative and are not restricted—feel free to invent engaging or imaginative labels.
3. The sequence must conclude with a "sleeping"-related activity, with the final timestamp corresponding to the next day’s wake-up time.
4. If the current time is in the evening or late at night, the first activity in the plan must also be a "sleeping"-related behavior. Otherwise, the first one must be the activity after waking up.
5. You should leave some time between activities for mobility or other things. Time ranges do not need to be continuous. 
6. This is a general and high-level daily plan—focus on key events and skip minor or trivial details. Generally, the time schedule should not be too busy.
7. **The schedule MUST reasonably conforms to the aforementioned context information: For example, if your age is not between 18-60, you should not go to work; your activity plan must be based on your internal requirements and external environments.
 In addition to being busy with your career and resting, you should generate as many activity needs as possible (e.g., sports, shopping, seeing a dentist, taking a walk, dating...) to make your life meaningful.**

Think outside the box, generate your meaningful daily plan! 

Please response in json format (Do not return any other text), example:
{{
    "daily_plan": [
        {{
            "behavior": "sleep",
            "time_range": "00:00-07:00"
        }},
        {{
            "behavior": "have breakfast",
            "time_range": "07:15-07:30"
        }},
        {{
            "behavior": "go running",
            "time_range": "08:00-08:30"
        }},
        {{
            "behavior": "work",
            "time_range": "09:00-12:30"
        }},
        ......(fill in behaviors)
        {{
            "behavior": "sleep",
            "time_range": "22:50-06:50"
        }},
    ]
}}
All output text values must be in English.
"""


class PlanBlock(Block):
    """A block for generating and managing execution plans through LLM-guided decision making.

    Attributes:
        configurable_fields: List of configurable parameter names
        default_values: Default values for configurable parameters
        fields_description: Human-readable descriptions for configurable parameters
        guidance_options: Predefined options mapped to specific needs
        max_plan_steps: Maximum allowed steps in generated plans (configurable)
    """

    def __init__(
        self,
        agent: Agent,
        toolbox: AgentToolbox,
        agent_memory: Memory,
        agent_context: DotDict,
        max_plan_steps: int = 5,
        detailed_plan_prompt: str = DETAILED_PLAN_PROMPT,
    ):
        """Initialize PlanBlock with required components.

        Args:
            llm: Language Model interface for decision making
            environment: Environment for contextual data
            memory: Agent's memory storage for status tracking
        """
        super().__init__(toolbox=toolbox, agent_memory=agent_memory)
        self.set_agent(agent)
        self.context = agent_context
        self.guidance_prompt = FormatPrompt(template=DETAILED_PLAN_PROMPT)
        self.daily_plan_prompt = FormatPrompt(
            template=DAILY_PLAN_PROMPT, memory=agent_memory
        )
        self.trigger_time = 0
        self.token_consumption = 0
        self.guidance_options = {
            "hungry": ["eat at home", "eat outside", "eat at current location", "eat nearby"],
            "tired": ["sleep at home"],
            # "safe": ["Work", "Shopping"],
            "social": ["contact with friends"],
            # "whatever": ["leisure and entertainment", "other", "stay at home"],
        }

        self.context["max_plan_steps"] = max_plan_steps

    def _coerce_minutes_str(self, value: Any, default: str = "10") -> str:
        if isinstance(value, (int, float)):
            return str(max(1, int(round(float(value)))))
        if isinstance(value, str):
            text = value.strip()
            if text.upper() == "TBD":
                return "TBD"
            m = re.search(r"-?\d+(\.\d+)?", text)
            if m:
                try:
                    return str(max(1, int(round(float(m.group(0))))))
                except Exception:
                    return default
        return default

    def _is_valid_time_range(self, value: Any) -> bool:
        return isinstance(value, str) and bool(
            re.match(r"^\d{2}:\d{2}\s*-\s*\d{2}:\d{2}$", value.strip())
        )

    async def select_guidance(self, current_need: str) -> Optional[Tuple[dict, str]]:
        """Select optimal guidance option using Theory of Planned Behavior evaluation.

        Args:
            current_need: The agent's current need to fulfill

        Returns:
            Optional[tuple[dict, str]]: Selected option with TPB evaluation scores and reasoning. None if no guidance option is selected by bad response from LLM.
        """
        cognition = None
        position_now = await self.memory.status.get("position")
        home_location = await self.memory.status.get("home")
        work_location = await self.memory.status.get("work")
        location_knowledge = await self.memory.status.get("location_knowledge")
        known_locations = [item["id"] for item in location_knowledge.values()]
        #assert len(known_locations) == 0
        id_to_name = {
            info["id"]: f"{name}({info['description']})"
            for name, info in location_knowledge.items()
        }
        current_location = "Outside"
        if (
            "aoi_position" in position_now
            and position_now["aoi_position"] == home_location["aoi_position"]
        ):
            current_location = "At home"
        elif (
            "aoi_position" in position_now
            and position_now["aoi_position"] == work_location["aoi_position"]
        ):
            current_location = "At workplace"
        elif (
            "aoi_position" in position_now
            and position_now["aoi_position"] in known_locations
        ):
            current_location = id_to_name[position_now["aoi_position"]]
        # else:
        #     assert "aoi_position" in position_now
        #     CUR_AOI = self.environment.map.get_aoi(position_now["aoi_position"]["aoi_id"])
        #     current_location = f'At {CUR_AOI["name"]}'
        
        day, current_time = self.environment.get_datetime(format_time=True)
        options = self.guidance_options.get(current_need, [])
        if len(options) == 0:
            options = "N/A"
        await self.guidance_prompt.format(
            current_need=current_need if len(options) > 0 else "N/A",
            weather=self.environment.sense("weather"),
            temperature=self.environment.sense("temperature"),
            other_info=self.environment.sense("other_information"),
            options=options,
            current_location=current_location,
            current_time=current_time,
            consumption_level=await self.memory.status.get("consumption"),
            occupation=await self.memory.status.get("occupation"),
            age=await self.memory.status.get("age"),
            emotion_types=await self.memory.status.get("emotion_types"),
            thought=await self.memory.status.get("thought"),
            daily_plan=self.context["daily_plan"],
            daily_plan_histories=self.context["daily_plan_histories"],
            max_plan_steps=self.context["max_plan_steps"]
        )

        retry = 15
        while retry > 0:
            try:
                response = await self.llm.atext_request(
                    self.guidance_prompt.to_dialog(), response_format={"type": "json_object"}
                )
                result: Any = json_repair.loads(clean_json_response(response))
                if "activity" not in result or "evaluation" not in result:
                    raise ValueError("Invalid guidance selection format")
                if (
                    "attitude" not in result["evaluation"]
                    or "subjective_norm" not in result["evaluation"]
                    or "perceived_control" not in result["evaluation"]
                    or "reasoning" not in result["evaluation"]
                ):
                    raise ValueError(
                        "Evaluation must include attitude, subjective_norm, perceived_control, and reasoning"
                    )
                if (
                    "activity_steps" not in result
                ):
                    raise ValueError("Invalid plan format")
                if not isinstance(result["activity_steps"], list):
                    raise ValueError("activity_steps must be a list")
                for step in result["activity_steps"]:
                    if "intention" not in step or "type" not in step or "duration" not in step or "subtype" not in step:
                        raise ValueError("Each step must have an intention and a type")   
                    step["type"] = str(step["type"]).lower()
                    if step["type"] not in ["mobility", "social", "other"]:
                        # Debug print removed.
                        # raise ValueError("type out of range")
                        step["type"] = "other"
                    step["subtype"] = str(step["subtype"]).lower()
                    if step["type"] == "other" and step["subtype"] not in ["sleeping", "eating", "working", "studying", "entertainment", "sporting", "trivial"]:
                        # Debug print removed.
                        # raise ValueError("other subtype out of range")
                        step["subtype"] = "trivial"
                    step["intention"] = str(step["intention"]).strip() or "do something"
                    if step["type"] == "mobility" and str(step["duration"]).strip().upper() != "TBD":
                        # Debug print removed.
                        step["duration"] = "TBD"
                        #raise ValueError("mobility must be TBD")
                    if step["type"] != "mobility":
                        normalized_duration = self._coerce_minutes_str(step["duration"], "10")
                        if normalized_duration == "TBD":
                            normalized_duration = "10"
                        if str(step["duration"]) != normalized_duration:
                            # Debug print removed.
                            pass
                        step["duration"] = normalized_duration
                guidance_result = {"activity": result["activity"], "evaluation": result["evaluation"]}
                cognition = f"I choose to {result['activity']} because {result['evaluation']['reasoning']}"
                detailed_plan = result["activity_steps"]
                return guidance_result, cognition, detailed_plan
            except Exception as e:
                get_logger().warning(
                    f"Error parsing guidance selection response: {str(e)}"
                )
                retry -= 1
        return None

    async def generate_daily_plan(self):
        
        await self.daily_plan_prompt.format(
            context=self.context,
        )

        retry = 15
        response = ""
        while retry > 0:
            try:
                response = await self.llm.atext_request(self.daily_plan_prompt.to_dialog())
                result: Any = json_repair.loads(clean_json_response(response))
                if (
                    "daily_plan" not in result
                ):
                    raise ValueError("Invalid plan format")
                for step in result["daily_plan"]:
                    if "behavior" not in step or "time_range" not in step:
                        raise ValueError("Each step must have abehavior and a time_range")
                    step["behavior"] = str(step["behavior"]).strip() or "unknown"
                    if not self._is_valid_time_range(step["time_range"]):
                        raise ValueError(f"Invalid time_range format: {step['time_range']}")
                return result["daily_plan"]
            except Exception as e:
                get_logger().warning(
                    f"Error parsing detailed plan: {str(e)} with response={response}"
                )
                retry -= 1
        return None
        
    async def forward(self):
        """Main workflow: Guidance selection -> Plan generation -> Memory update"""
        
        if self.context["daily_plan_activation"]:
            daily_plan = None
            for _ in range(5):
                try:
                    daily_plan = await self.generate_daily_plan()
                    if daily_plan is None:
                        continue
                    break
                except:
                    pass
            if daily_plan is None:
                daily_plan = [{"behavior": "sleep", "time_range": "00:00-08:00"}]

            self.context["daily_plan"] = daily_plan
            await self.memory.status.update("current_daily_plan", daily_plan)
            self.context["daily_plan_histories"] = []
            await self.memory.status.update("current_daily_plan_histories", [])
            self.context["daily_plan_activation"] = False
        
        cognition = None
        current_need = await self.memory.status.get("current_need")
        select_guidance = None
        for _ in range(5):
            try:
                select_guidance = await self.select_guidance(current_need)
                if select_guidance is not None:
                    break
            except Exception as e:
                get_logger().warning(f"select_guidance failed: {type(e).__name__}: {e}")
        if select_guidance is None:
            select_guidance = (
                {
                    "activity": "do something nearby",
                    "evaluation": {
                        "attitude": "0.5",
                        "subjective_norm": "0.5",
                        "perceived_control": "0.5",
                        "reasoning": "fallback due to repeated parse errors",
                    },
                },
                "Fallback plan due to repeated parsing errors.",
                [
                    {
                        "intention": "do something nearby",
                        "type": "other",
                        "subtype": "trivial",
                        "duration": "10",
                    }
                ],
            )
        # Debug print removed.

        guidance_result, cognition, detailed_plan = select_guidance
        
        # guidance_result, cognition, detailed_plan = \
        #     {'activity': 'go working', 'evaluation': {'attitude': '0.9', 'subjective_norm': '0.8', 'perceived_control': '0.95', 'reasoning': "None"}},\
        #     "I choose to work because it is a straightforward task that can be easily controlled.", \
        #     [{'intention': 'go to work', 'type': 'mobility', 'subtype': 'mobility', 'duration': '15'}]
        # guidance_result, cognition, detailed_plan = \
        #     {'activity': 'have breakfast', 'evaluation': {'attitude': '0.9', 'subjective_norm': '0.8', 'perceived_control': '0.95', 'reasoning': "None"}},\
        #     "I choose to have breakfast because it is a straightforward task that can be easily controlled.", \
        #     [{'intention': 'have breakfast', 'type': 'other', 'subtype': 'eating', 'duration': '3'}]
        # guidance_result, cognition, detailed_plan = \
        #     {'activity': 'chat', 'evaluation': {'attitude': '0.9', 'subjective_norm': '0.8', 'perceived_control': '0.95', 'reasoning': "None"}},\
        #     "I choose to have breakfast because it is a straightforward task that can be easily controlled.", \
        #     [{'intention': 'chat', 'type': 'social', 'subtype': 'social', 'duration': '3'}]
                
        self.context["plan_target"] = guidance_result["activity"]
        self.context["daily_plan_histories"].append(self.context["plan_target"])
        await self.memory.status.update("current_daily_plan_histories", self.context["daily_plan_histories"])
        # Debug print removed.

        # Step 3: Update plan and current step
        steps = detailed_plan
        for step in steps:
            step["evaluation"] = {"status": "pending", "details": ""}

        plan = {
            "target": guidance_result["activity"],
            "steps": steps,
            "index": 0,
            "completed": False,
            "failed": False,
            "stream_nodes": [],
            "guidance": guidance_result,  # Save the evaluation result of the plan selection
        }
        formated_steps = "\n".join(
            [f"{i}. {step['intention']}" for i, step in enumerate(plan["steps"], 1)]
        )
        formated_plan = f"""
Overall Target: {plan['target']}
Execution Steps: \n{formated_steps}
        """
        _, plan["start_time"] = self.environment.get_datetime(format_time=True)
        await self.memory.status.update("current_plan", plan)
        await self.memory.status.update("execution_context", {"plan": formated_plan})
        await self.memory.stream.add(
            topic="cognition",
            description=cognition,
        )
        position_now = await self.memory.status.get("position")
        home_location = await self.memory.status.get("home")
        work_location = await self.memory.status.get("work")
        if "aoi_position" in position_now:
            if position_now["aoi_position"] == home_location["aoi_position"]:
                self.context.current_plan_related_places = [(position_now["aoi_position"]["aoi_id"], "home")]
            elif position_now["aoi_position"] == work_location["aoi_position"]:
                self.context.current_plan_related_places = [(position_now["aoi_position"]["aoi_id"], "workplace")]
            else:
                self.context.current_plan_related_places = [(position_now["aoi_position"]["aoi_id"],
                                                    self.environment.map.get_aoi(position_now["aoi_position"]["aoi_id"])["name"])]
        return cognition
