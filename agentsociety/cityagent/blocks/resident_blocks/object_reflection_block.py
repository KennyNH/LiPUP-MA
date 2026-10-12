from typing import Any, Optional
import json_repair
from pydantic import Field
from ....logger import get_logger
from ....memory import Memory
from ....agent import AgentToolbox, Block, FormatPrompt, BlockParams, DotDict
from landtype2dec import type2dec_dict_HK

__all__ = ["ObjectReflectionBlock"]


def extract_json(output_str):
    """Extract JSON substring from a raw string response.

    Args:
        output_str: Raw string output that may contain JSON data.

    Returns:
        Extracted JSON string if valid, otherwise None.

    Note:
        Searches for the first '{' and last '}' to isolate JSON content.
        Catches JSON decoding errors and logs warnings.
    """
    try:
        # Find the positions of the first '{' and the last '}'
        start = output_str.find("{")
        end = output_str.rfind("}")

        # Extract the substring containing the JSON
        json_str = output_str[start : end + 1]

        # Convert the JSON string to a dictionary
        return json_str
    except ValueError as e:
        get_logger().warning(f"Failed to extract JSON: {e}")
        return None


class ObjectReflectionBlockParams(BlockParams):
    top_k: int = Field(
        default=20, description="Number of most relevant memories to return"
    )
    
OBJECT_REFLECTION_PROMPT = """

Current weather: {weather}
Current temperature: {temperature}
Current location: {current_location}
Current time: {current_time}

You are a {gender}, aged {age}, belonging to the {race} race and identifying as {religion}. 
Your marital status is {marriage_status}, and you currently reside in a {residence} area. 
Your occupation is {occupation}, and your education level is {education}. 
You are {personality}, with a consumption level of {consumption} and a family consumption level of {family_consumption}. 
Your income is {income}, and you are skilled in {skill}.
My current emotion intensities are (0 meaning not at all, 10 meaning very much):
sadness: {sadness}, joy: {joy}, fear: {fear}, disgust: {disgust}, anger: {anger}, surprise: {surprise}.
You have the following thoughts: {thought}.
In the following 21 words, I have chosen {emotion_types} to represent your current status:
Joy, Distress, Resentment, Pity, Hope, Fear, Satisfaction, Relief, Disappointment, Pride, Admiration, Shame, Reproach, Liking, Disliking, Gratitude, Anger, Gratification, Remorse, Love, Hate.

{incidents}

You have just completed an activity, with the details provided below:
{current_plan_info}

I am interested in your feelings, judgments, or comments/suggestions regarding the urban planning (land-use) quality of the places you visited recently: {visited_places}. 
You can integrate the information of these places and the activity details together for analysis and give your feedbacks.
We are collecting residents' opinions to support the renewal of urban functional areas. You can pose your satisfaction/dissatisfaction points, or comments/suggestions for the renewal.

**Your reflection/feedbacks SHOULD MUST take full advantage of the above activity information, especially mobility-related activity, because it is important for us to improve your service accessibility.
  For example, normally an urban functional area needs to be accessible to people within 500 meters and 15 minutes, any number exceeding the range is unreasonable. 
  You may flexibly decide your tolerance based on your own information(note that the units of "consumed_time" and "distance_m" in the activity details are minute and meter, respectively.
  You can also reflect on other factors, or other activity type. Please think OUTSIDE THE BOX, but MUST BE REASONABLE and fully consider your past information.
  I suggest you to reflect from two aspects:
  1. For some places themselves, you may reflect on whether the spatial distribution/land-use is planned to satisfy your preference and requirements.
  2. For the nearby planning of some places, you can state your feeling, or propose your idea (e.g. If you have a long walk from home to the gym every time, you might be dissatisfied, and wish a gym near your home).
  **

You have previously shared some observations on these locations:
{past_comments}

Please feel free to update, remove, or add to your previous impressions based on your latest experience. 
Just give 1-2 comments for each area that you think deserve a review. You'd better to divide comments according to the place itself/surroundings, as I mentioned.

Please response in json format (Do not return any other text, note that your key must be id ,not name), example:
{{
    "place_id": {{
        "comment_1": "xxx",
        "comment_2": "xxx",
    }},
    "place_id": {{
        "comment_1": "xxx",
        "comment_2": "xxx",
    }}
}}
All output text values must be in English.
"""

class ObjectReflectionBlock(Block):

    ParamsType = ObjectReflectionBlockParams
    name = "ObjectReflectionBlock"
    description = "Handles daily updates of attitudes, thoughts, and emotions"
    actions = {}

    def __init__(
        self,
        toolbox: AgentToolbox,
        agent_memory: Memory,
        agent_context: DotDict,
        block_params: Optional[ObjectReflectionBlockParams] = None,
    ):
        """Initialize ObjectReflectionBlock with dependencies.

        Args:
            llm: Language Model interface for cognitive processing.
            environment: Environment for time-based operations.
            memory: Memory system to store/retrieve agent status and experiences.
        """
        super().__init__(
            toolbox=toolbox,
            agent_memory=agent_memory,
            block_params=block_params,
        )
        self.last_check_day = None
        self.context = agent_context
        self.object_reflection_prompt = FormatPrompt(template=OBJECT_REFLECTION_PROMPT)
        
    async def memory_update(self):
        
        emotion = await self.memory.status.get("emotion")
        sadness = int(emotion.get("sadness", 5)) if isinstance(emotion, dict) else 5
        joy = int(emotion.get("joy", 5)) if isinstance(emotion, dict) else 5
        fear = int(emotion.get("fear", 5)) if isinstance(emotion, dict) else 5
        disgust = int(emotion.get("disgust", 5)) if isinstance(emotion, dict) else 5
        anger = int(emotion.get("anger", 5)) if isinstance(emotion, dict) else 5
        surprise = int(emotion.get("surprise", 5)) if isinstance(emotion, dict) else 5
        
        position_now = await self.memory.status.get("position")
        home_location = await self.memory.status.get("home")
        work_location = await self.memory.status.get("work")
        location_knowledge = await self.memory.status.get("location_knowledge")
        known_locations = [item["id"] for item in location_knowledge.values()]
        if len(known_locations) != 0:
            get_logger().warning(
                f"ObjectReflectionBlock: known_locations expected empty, got {len(known_locations)}"
            )
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
        else:
            # assert "aoi_position" in position_now
            # CUR_AOI = self.environment.map.get_aoi(position_now["aoi_position"]["aoi_id"])
            # current_location = f'At {CUR_AOI["name"]}'
            current_location="Outside"
        day, current_time = self.environment.get_datetime(format_time=True)
               
        incident_str = await self.memory.stream.search_today(top_k=20)
        if incident_str:
            incident_prompt = "Today, these incidents happened:\n" + incident_str
        else:
            incident_prompt = "No incidents happened today."
        
        visited_places = self._format_visited_places_with_landuse(
            self.context.current_plan_related_places
        )

        await self.object_reflection_prompt.format(
            current_location=current_location,
            current_time=current_time,
            weather=self.environment.sense("weather"),
            temperature=self.environment.sense("temperature"),
            gender=await self.memory.status.get("gender"),
            age=await self.memory.status.get("age"),
            race=await self.memory.status.get("race"),
            religion=await self.memory.status.get("religion"),
            marriage_status=await self.memory.status.get("marriage_status"),
            residence=await self.memory.status.get("residence"),
            occupation=await self.memory.status.get("occupation"),
            education=await self.memory.status.get("education"),
            personality=await self.memory.status.get("personality"),
            consumption=await self.memory.status.get("consumption"),
            family_consumption=await self.memory.status.get("family_consumption"),
            income=await self.memory.status.get("income"),
            skill=await self.memory.status.get("skill"),
            sadness=sadness,
            joy=joy,
            fear=fear,
            disgust=disgust,
            anger=anger,
            surprise=surprise,
            emotion=await self.memory.status.get("emotion"),
            thought=await self.memory.status.get("thought"),
            emotion_types=await self.memory.status.get("emotion_types"),
            incidents=incident_prompt,
            #current_plan_info=reshape_plan(await self.memory.status.get("current_plan", None)),
            current_plan_info=await self.memory.status.get("current_plan", None),
            visited_places=visited_places,
            past_comments=await self.memory.objmem.get_aoi_mem(self.context.current_plan_related_places)
        )




        expected_ids = {str(p[0]) for p in self.context.current_plan_related_places}
        for retry in range(10):
            try:
                _response = await self.llm.atext_request(
                    self.object_reflection_prompt.to_dialog(),
                    timeout=300,
                    response_format={"type": "json_object"},
                )
                json_str = extract_json(_response)
                if not json_str:
                    get_logger().warning("ObjectReflectionBlock: no JSON extracted from model response")
                    continue
                response: Any = json_repair.loads(json_str)
                if not isinstance(response, dict):
                    get_logger().warning(f"ObjectReflectionBlock: response must be dict, got {type(response)}")
                    continue

                normalized: dict[str, dict[str, str]] = {}
                for _ID, comments in response.items():
                    aoi_id = str(_ID)
                    if aoi_id not in expected_ids:
                        continue
                    if not isinstance(comments, dict):
                        get_logger().warning(
                            f"ObjectReflectionBlock: comments must be dict for AOI={aoi_id}, got={type(comments)}"
                        )
                        continue
                    clean_comments: dict[str, str] = {}
                    idx = 1
                    for _, v in comments.items():
                        txt = str(v).strip() if v is not None else ""
                        if not txt:
                            continue
                        clean_comments[f"comment_{idx}"] = txt
                        idx += 1
                    if clean_comments:
                        normalized[aoi_id] = clean_comments

                # strict on id scope; tolerant on empty comments (handled downstream)
                if not normalized:
                    get_logger().warning(
                        f"ObjectReflectionBlock retry {retry+1}/10: no valid reflection comments generated"
                    )
                    continue
                await self.memory.objmem.update_aoi_mem(normalized)
                return
            except Exception as e:
                get_logger().warning(
                    f"ObjectReflectionBlock retry {retry+1}/10 failed: {type(e).__name__}: {e}"
                )

        # Do not crash the simulation on reflection formatting failures.
        get_logger().warning("ObjectReflectionBlock: all retries failed; skip this reflection cycle")
        return

    async def forward(self):

        await self.memory_update()

    def _format_visited_places_with_landuse(self, visited_places):
        formatted = []
        for place in visited_places:
            if not isinstance(place, (list, tuple)) or len(place) < 2:
                continue
            aoi_id = place[0]
            place_name = place[1]
            land_use_full = "Unknown"
            try:
                aoi = self.environment.map.get_aoi(int(aoi_id))
                land_use_code = aoi.get("urban_land_use")
                land_use_full = type2dec_dict_HK.get(land_use_code, str(land_use_code))
            except Exception:
                pass
            formatted.append(
                {
                    "aoi_id": aoi_id,
                    "name": place_name,
                    "land_use": land_use_full,
                }
            )
        return formatted



def reshape_plan(plan):
    '''
    把plan喂给大模型可读性更强

    :param plan:
    :return:
    '''
    step_dec = []
    for i in range(len(plan["steps"])):
        step_info = plan["steps"][i]
        if step_info["type"] == "mobility":
            step_dec.append(f'''step_{i+1}: The intention is to **{step_info["intention"]}**, the behavior type is **{step_info["subtype"]}**, moving from position ({step_info["position"]}), and {step_info["evaluation"]["evaluation"]} after **{step_info["evaluation"]["consumed_time"]} minutes**.''')

        else:
            step_dec.append(f'''step{i+1}: The intention is to **{step_info["intention"]}**, the behavior type is **{step_info["subtype"]}**, at position ({step_info["position"]}), result: {step_info["evaluation"]["evaluation"]} after **{step_info["evaluation"]["consumed_time"]} minutes**.''')

    description = f'''The target of last activity plan is to **{plan['target']}**, start from **{plan['start_time']}**, the reason is: {plan["guidance"]["evaluation"]["reasoning"]}. The plan includes the following steps: {step_dec}'''

    return description
