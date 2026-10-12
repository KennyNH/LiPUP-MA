import math
import random
import re
from typing import Optional
from copy import deepcopy

import json_repair
import numpy as np
from pydantic import Field

from ....agent import (
    Block,
    FormatPrompt,
    BlockParams,
    DotDict,
    BlockContext,
    AgentToolbox,
)
from ....logger import get_logger
from ....memory import Memory
from ....agent.dispatcher import BlockDispatcher
from ...sharing_params import SocietyAgentMobilityBlockOutput
from ..utils import clean_json_response
from landtype2dec import *


# Prompt templates for LLM interactions
PLACE_TYPE_SELECTION_PROMPT = """
As an intelligent decision system, please determine the type of place the user needs to visit based on their input requirement.
User Plan: {plan}
User requirement: {intention}
Other information: 
-------------------------
{other_info}
-------------------------
Your output must be selected from {aoi_category} without any additional text or explanation. 
Select the candidate place categories, exclude the impossible ones and reserve the most related ones (1-3 categories).

If there's indeed no suitable choice, you can select "home" or "workplace". But you are encouraged to explore more urban space and activities, try not to choose these two.

Please response in json format (Do not return any other text), example:
{{
    "candidate_place_types": ["category 1", "category 2", "category 3"]
}}
All output text values must be in English.
"""

PLACE_NAME_SELECTION_PROMPT = """
As an intelligent decision system, please determine the type of place the user needs to visit based on their input requirement.
User Plan: {plan}
User requirement: {intention}
Other information: 
-------------------------
{other_info}
-------------------------
Your output must be selected from the candidate place information below, without any additional text or explanation.
{aoi_info}

Select the candidate places, exclude the impossible ones and reserve the most user-preferred ones (no number limit, but should be reasonable).
If a place name is absent, it should be reserved. You can make your choice just based on the land-use category or randomly.
Just answer the [place id, place name] without any detailed description.

Please response in json format (Do not return any other text), example:
{{
    "candidate_places": [["id 1", "name 1"], ["id 2", "name 2"], ["id 3", "name 3"], ["id 4", "name 4"], ["id 5", "name 5"]]
}}
For "name", you MUST copy the exact place name from the candidate list (do NOT translate, summarize, or replace with land-use category words).
"""

RADIUS_PROMPT = """As an intelligent decision system, please determine the maximum travel radius (in meters) based on the current emotional state.

Current weather: ${context.weather}
Current temperature: ${context.temperature}
Your current emotion: ${context.current_emotion}
Your current thought: ${context.current_thought}
Other information: 
-------------------------
${context.other_information}
-------------------------

Please analyze how these emotions would affect travel willingness and return only a single integer number between 3000-200000 representing the maximum travel radius in meters. A more positive emotional state generally leads to greater willingness to travel further.

Please response in json format (Do not return any other text), example:
{{
    "radius": 10000
}}
"""


AOI_CATE_DEC=type2dec_dict_HK

AOI_CATE_DEC_REVERSE = {v: k for k, v in AOI_CATE_DEC.items()}
##这个用来写HKWJ,只针对HK地图

def _coerce_int(value, default: int, min_v: Optional[int] = None, max_v: Optional[int] = None) -> int:
    try:
        if isinstance(value, str):
            value = value.strip()
        v = int(round(float(value)))
    except Exception:
        v = default
    if min_v is not None:
        v = max(min_v, v)
    if max_v is not None:
        v = min(max_v, v)
    return v


def _extract_candidate_places(parsed: object) -> list:
    """Normalize model output into a candidate place list.

    Accepts common variants:
    - {"candidate_places": [...]}
    - {"candidate_place": [...]}
    - {"candidates": [...]}
    - {"places": [...]}
    - a direct list
    - string payload that itself encodes JSON/list/ids
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        raw = (
            parsed.get("candidate_places")
            or parsed.get("candidate_place")
            or parsed.get("candidates")
            or parsed.get("places")
        )
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            # Try parse nested JSON/list string.
            try:
                nested = json_repair.loads(clean_json_response(raw))
                if isinstance(nested, list):
                    return nested
                if isinstance(nested, dict):
                    return _extract_candidate_places(nested)
            except Exception:
                # Last-resort: extract numeric ids from text.
                nums = re.findall(r"-?\d+", raw)
                if nums:
                    return [int(x) for x in nums]
    return []


def gravity_model(aois): #这里逻辑要不要改：怎么去判断AOI的吸引力？

    # Initialize distance bins
    aois_Dis = {f"{d}k": [] for d in range(1, 11)}
    aois_Dis["more"] = []

    # Categorize POIs into distance bins
    for aoi in aois:
        classified = False
        for d in range(1, 11):
            if (d - 1) * 1000 <= aoi[1] < d * 1000:
                aois_Dis[f"{d}k"].append(aoi)
                classified = True
                break
        if not classified:
            aois_Dis["more"].append(aoi)

    res = []
    distanceProb = []
    # Calculate weights for each POI
    for aoi in aois:
        for d in range(1, 11):
            if (d - 1) * 1000 <= aoi[1] < d * 1000:
                n = len(aois_Dis[f"{d}k"])
                # Calculate ring area between (d-1)km and d km
                S = math.pi * ((d * 1000) ** 2 - ((d - 1) * 1000) ** 2)
                density = n / S  # POIs per square meter
                distance = max(aoi[1], 1)  # Avoid division by zero

                # Inverse square distance decay combined with density
                weight = density / (distance**2)
                res.append((aoi[0]["name"], aoi[0]["id"], weight, distance))
                distanceProb.append(1 / math.sqrt(distance))
                break

    # Normalize probabilities and sample
    distanceProb = np.array(distanceProb)
    distanceProb /= distanceProb.sum()

    # Randomly sample 50 candidates weighted by distance probabilities
    sample_indices = np.random.choice(len(res), size=50, p=distanceProb)
    sampled_aois = [res[i] for i in sample_indices]

    # Normalize weights for final selection
    total_weight = sum(item[2] for item in sampled_aois)
    return [
        (item[0], item[1], item[2] / total_weight, item[3]) for item in sampled_aois
    ]


class PlaceSelectionBlock(Block):
    """
    Block for selecting destinations based on user intention.

    Implements a two-stage selection process:
    1. Select primary POI category (e.g., 'shopping')
    2. Select sub-category (e.g., 'bookstore')
    Uses LLM for decision making with fallback to random selection.

    Configurable Fields:
        search_limit: Max number of POIs to retrieve from map service
    """

    name = "PlaceSelectionBlock"
    description = "Selects destinations for unknown locations (excluding home/work)"

    def __init__(
        self,
        toolbox: AgentToolbox,
        agent_memory: Memory,
        search_limit: int = 50,
    ):
        super().__init__(
            toolbox=toolbox,
            agent_memory=agent_memory,
        )
        self.typeSelectionPrompt = FormatPrompt(PLACE_TYPE_SELECTION_PROMPT)
        self.nameSelectionPrompt = FormatPrompt(PLACE_NAME_SELECTION_PROMPT)

        self.radiusPrompt = FormatPrompt(
            RADIUS_PROMPT,
            memory=agent_memory,
        )
        self.search_limit = search_limit  # Default config value
    
    def get_aoi_categories(self):
        #如果有缺少的，直接用字母作为描述
        L = deepcopy(AOI_CATE_DEC)
        
        map_aoi_list = self.environment.map._aoi_list
        type_list = [a['urban_land_use'] for a in map_aoi_list]
        for t in type_list:
            if t not in L:
                L[t] = t
        
        return list(L.values())

    async def forward(self, context: DotDict):
        """Execute the destination selection workflow"""

        aoi_category = self.get_aoi_categories()
        NotFound = False
        AOI_LAND_TYPE = []

        # Select AOI category
        await self.typeSelectionPrompt.format(
            plan=context["plan_context"]["plan"],
            intention=context["current_step"]["intention"],
            aoi_category=aoi_category,
            other_info=self.environment.environment.get("other_information", "None"),
        )
        try:
            retry = 0
            while retry < 10:
                # LLM-based category selection
                AOI_LAND_TYPE = await self.llm.atext_request(
                    self.typeSelectionPrompt.to_dialog(),
                    response_format={"type": "json_object"},
                )
                AOI_LAND_TYPE = json_repair.loads(clean_json_response(AOI_LAND_TYPE))["candidate_place_types"] # type: ignore
                if not isinstance(AOI_LAND_TYPE, list):
                    retry += 1
                    get_logger().warning(f"MobilityBlock: invalid candidate_place_types type={type(AOI_LAND_TYPE)}")
                    continue
                if all(i in aoi_category+["home","workplace"] for i in AOI_LAND_TYPE):
                    break
                retry += 1
            if not isinstance(AOI_LAND_TYPE, list):
                AOI_LAND_TYPE = []
            if "home" in [t.lower() for t in AOI_LAND_TYPE]:
                context["next_place"] = "home"
                return "home", NotFound
            elif "workplace" in [t.lower() for t in AOI_LAND_TYPE]:
                context["next_place"] = "workplace"
                return "workplace", NotFound
        except Exception as e:
            get_logger().warning(f"MobilityBlock: Level 1 selection failed: {e}")
        if not AOI_LAND_TYPE:
            AOI_LAND_TYPE = [random.choice(aoi_category)]

        # Get travel radius from LLM
        try:
            await self.radiusPrompt.format(context=context)
            radius = await self.llm.atext_request(
                self.radiusPrompt.to_dialog(), response_format={"type": "json_object"}
            )
            radius = _coerce_int(json_repair.loads(radius).get("radius"), 10000, 3000, 200000) # type: ignore
        except Exception as e:
            get_logger().warning(f"MobilityBlock: Radius selection failed: {e}")
            radius = 10000  # Default 10km

        # Query and select POI
        xy = (await self.memory.status.get("position"))["xy_position"]
        center = (xy["x"], xy["y"])
        aois_queried = self.environment.map.query_aois(
            center=center,
            urban_land_uses=[AOI_CATE_DEC_REVERSE[t] for t in AOI_LAND_TYPE],
            radius=radius,
            limit=self.search_limit,
        )
        if len(aois_queried) == 0:
            aois_queried = self.environment.map.query_aois(
                center=center,
                urban_land_uses=[AOI_CATE_DEC_REVERSE[t] for t in AOI_LAND_TYPE],
                radius=radius,
                limit=self.search_limit,
            )
            NotFound = True
        if len(aois_queried) == 0:
            # No candidates after query/retry: fall back directly to all AOIs to avoid
            # repeatedly calling LLM with empty candidate list.
            all_aois = self.environment.map.get_all_aois()
            if all_aois:
                picked = random.choice(all_aois)
                next_place = (picked.get("name", ""), picked.get("id"))
                context["next_place"] = next_place
                return next_place, True
            # Degenerate map case: return home to keep simulation alive.
            context["next_place"] = "home"
            return "home", True
        
        # Select AOI name
        aoi_info = [f"id: {tmp_i}, name: {n[0]['name']}, category: {AOI_CATE_DEC[n[0]['urban_land_use']]}" for tmp_i, n in enumerate(aois_queried)]
        await self.nameSelectionPrompt.format(
            plan=context["plan_context"]["plan"],
            intention=context["current_step"]["intention"],
            aoi_info=aoi_info,
            other_info=self.environment.environment.get("other_information", "None"),
        )
        aois = []
        try:
            # LLM-based category selection
            AOI_NAME = await self.llm.atext_request(
                self.nameSelectionPrompt.to_dialog(),
                response_format={"type": "json_object"},
            )
            parsed = json_repair.loads(clean_json_response(AOI_NAME))
            AOI_NAME = _extract_candidate_places(parsed)
            if not AOI_NAME:
                # Strong fallback: use top candidates directly instead of raising.
                AOI_NAME = list(range(min(len(aois_queried), 5)))
            for item in AOI_NAME:
                if isinstance(item, (int, str)):
                    _ID, _NAME = item, ""
                elif isinstance(item, dict):
                    _ID = item.get("id", item.get("index", item.get("aoi_index")))
                    _NAME = item.get("name", "")
                elif isinstance(item, (list, tuple)) and len(item) >= 1:
                    _ID = item[0]
                    _NAME = item[1] if len(item) >= 2 else ""
                else:
                    get_logger().warning(f"MobilityBlock: invalid candidate_place item format: {item}")
                    continue

                #print("id ",_ID,"name ",_NAME)
                if not isinstance(_ID, int):
                    try:
                        _ID = int(_ID)
                    except Exception:
                        # Accept variants like "id 0" / "ID:12"
                        m = re.search(r"-?\d+", str(_ID))
                        if m:
                            try:
                                _ID = int(m.group(0))
                            except Exception:
                                get_logger().warning(f"MobilityBlock: invalid AOI index type={type(_ID)} value={_ID}")
                                continue
                        else:
                            get_logger().warning(f"MobilityBlock: invalid AOI index type={type(_ID)} value={_ID}")
                            continue
                if _ID < 0 or _ID >= len(aois_queried):
                    get_logger().warning(f"MobilityBlock: AOI index out of range: {_ID}")
                    continue
                map_name = str(aois_queried[_ID][0].get("name") or "")
                llm_name = str(_NAME or "")
                category_words = {str(v).strip().lower() for v in AOI_CATE_DEC.values()}
                if map_name.strip().lower() != llm_name.strip().lower() and llm_name.strip():
                    # If model returns a category word (e.g., "Park", "Residence"), treat as weak name and skip warning.
                    if llm_name.strip().lower() in category_words:
                        aois.append(aois_queried[_ID])
                        continue
                    get_logger().warning(
                        f"MobilityBlock: AOI name mismatch at {_ID}, llm={llm_name}, map={map_name}"
                    )
                    # Keep index-valid candidates even when names mismatch.
                aois.append(aois_queried[_ID])
        except Exception as e:
            get_logger().warning(f"MobilityBlock: Level 1 selection failed: {e}")
            # LLM parse/runtime failure fallback: keep mobility alive with first few candidates.
            for i in range(min(len(aois_queried), 5)):
                aois.append(aois_queried[i])

        # gravity model-based sampling
        # assert aois
        if aois:
            aois = gravity_model(aois)
            probabilities = [item[2] for item in aois]
            selected = np.random.choice(len(aois), p=probabilities)
            next_place = (aois[selected][0], aois[selected][1])
        else:
            all_aois = self.environment.map.get_all_aois()
            next_place = random.choice(all_aois)
            next_place = (next_place["name"], next_place["id"])
        context["next_place"] = next_place
        return next_place, NotFound


class MoveBlock(Block):
    """Block for executing mobility operations (home/work/other)"""

    name = "MoveBlock"
    description = "Executes mobility operations between locations, especially for home, workplace and other known places"

    def __init__(self, toolbox: AgentToolbox, agent_memory: Memory):
        super().__init__(
            toolbox=toolbox,
            agent_memory=agent_memory,
        )

    async def _safe_set_aoi_schedule(self, person_id: int, target_aoi_id: int, now_place: dict):
        # The backend rejects SetSchedule when the person is currently in junction/lane state.
        if "aoi_position" not in now_place:
            return False, "position_not_in_aoi"
        try:
            await self.environment.set_aoi_schedules(
                person_id=person_id,
                target_positions=target_aoi_id,
            )
            return True, ""
        except Exception as e:
            err = str(e).lower()
            if "person in a junction" in err:
                # Lightweight one-time retry for transient state races.
                try:
                    await self.environment.set_aoi_schedules(
                        person_id=person_id,
                        target_positions=target_aoi_id,
                    )
                    return True, ""
                except Exception as e2:
                    return False, f"junction_retry_failed:{type(e2).__name__}"
            raise

    async def forward(self, next_place, context, NotFound=False):
        agent_id = await self.memory.status.get("id")
        place_knowledge = await self.memory.status.get("location_knowledge")
        known_places = list(place_knowledge.keys())
        if len(known_places) != 0:
            get_logger().warning(f"MoveBlock: known_places expected empty, got {len(known_places)}")
        if next_place != context.next_place:
            get_logger().warning(f"MoveBlock: next_place mismatch: {next_place} vs {context.next_place}")
        
        if next_place == "home":
            if NotFound:
                get_logger().warning("MoveBlock: NotFound=True when returning home")
            # go back home
            home = await self.memory.status.get("home")
            home = home["aoi_position"]["aoi_id"]
            nowPlace = await self.memory.status.get("position")
            if ("aoi_position" in nowPlace and nowPlace["aoi_position"]["aoi_id"] == home):
                return {
                    "success": True,
                    "evaluation": "Successfully returned home (already at home)",
                    "to_place": home,
                    "consumed_time": -10,
                    "node_id": None,
                }
            node_id = await self.memory.stream.add(
                topic="mobility",
                description="I returned home",
            )
            scheduled, reason = await self._safe_set_aoi_schedule(agent_id, home, nowPlace)
            if not scheduled:
                return {
                    "success": False,
                    "evaluation": f"Move deferred: cannot set schedule while not in AOI ({reason})",
                    "to_place": home,
                    "consumed_time": 1,
                    "node_id": None,
                }
            number_poi_visited = await self.memory.status.get("number_poi_visited")
            number_poi_visited += 1
            await self.memory.status.update("number_poi_visited", number_poi_visited)
            return {
                "success": True,
                "evaluation": "Successfully returned home",
                "to_place": home,
                "consumed_time": -10,
                "node_id": node_id,
            }
        elif next_place == "workplace":
            if NotFound:
                get_logger().warning("MoveBlock: NotFound=True when returning workplace")
            # back to workplace
            work = await self.memory.status.get("work")
            work = work["aoi_position"]["aoi_id"]
            nowPlace = await self.memory.status.get("position")
            if ("aoi_position" in nowPlace and nowPlace["aoi_position"]["aoi_id"] == work):
                return {
                    "success": True,
                    "evaluation": "Successfully reached the workplace (already at the workplace)",
                    "to_place": work,
                    "consumed_time": -10,
                    "node_id": None,
                }
            node_id = await self.memory.stream.add(
                topic="mobility",
                description="I went to my workplace",
            )
            scheduled, reason = await self._safe_set_aoi_schedule(agent_id, work, nowPlace)
            if not scheduled:
                return {
                    "success": False,
                    "evaluation": f"Move deferred: cannot set schedule while not in AOI ({reason})",
                    "to_place": work,
                    "consumed_time": 1,
                    "node_id": None,
                }
            number_poi_visited = await self.memory.status.get("number_poi_visited")
            number_poi_visited += 1
            await self.memory.status.update("number_poi_visited", number_poi_visited)
            return {
                "success": True,
                "evaluation": "Successfully reached the workplace",
                "to_place": work,
                "consumed_time": -10,
                "node_id": node_id,
            }
        else:
            # move to other places
            nowPlace = await self.memory.status.get("position")
            # Debug print removed.
            if "aoi_position" in nowPlace:
                nowPlace_aoi_id = nowPlace["aoi_position"]["aoi_id"]

                node_id = await self.memory.stream.add(
                    topic="cognition",
                    description=f"I'm now at {(nowPlace_aoi_id, self.environment.map.get_aoi(nowPlace_aoi_id)['name'])} and I cannot find a place that satisfies my current needs, thus I just try to find the most suitable alternative place: {next_place}",
                )

            if ("aoi_position" in nowPlace and nowPlace_aoi_id == next_place[1]):
                return {
                    "success": True,
                    "evaluation": f"Successfully reached the destination: {next_place} (already at {next_place})",
                    "to_place": next_place[1],
                    "consumed_time": -10,
                    "node_id": None,
                }
            node_id = await self.memory.stream.add(
                topic="mobility",
                description=f"I went to {next_place}",
            )
            scheduled, reason = await self._safe_set_aoi_schedule(agent_id, next_place[1], nowPlace)
            if not scheduled:
                return {
                    "success": False,
                    "evaluation": f"Move deferred: cannot set schedule while not in AOI ({reason})",
                    "to_place": next_place[1],
                    "consumed_time": 1,
                    "node_id": None,
                }
            number_poi_visited = await self.memory.status.get("number_poi_visited")
            number_poi_visited += 1
            await self.memory.status.update("number_poi_visited", number_poi_visited)
            return {
                "success": True,
                "evaluation": f"Successfully reached the destination: {next_place}",
                "to_place": next_place[1],
                "consumed_time": -10,
                "node_id": node_id,
            }


class MobilityBlockParams(BlockParams):
    # PlaceSelection
    radius_prompt: str = Field(
        default=RADIUS_PROMPT, description="Used to determine the maximum travel radius"
    )
    search_limit: int = Field(
        default=50, description="Number of POIs to retrieve from map service"
    )


class MobilityBlockContext(BlockContext):
    next_place: Optional[tuple[str, int]] = Field(
        default=None, description="The next place to go"
    )


class MobilityBlock(Block):
    """
    Main mobility coordination block.
    """

    ParamsType = MobilityBlockParams
    OutputType = SocietyAgentMobilityBlockOutput
    ContextType = MobilityBlockContext
    name = "MobilityBlock"
    description = "Used for moving like go to work, go to home, go to other places, etc."
    actions = {
        "place_selection": "Support the place selection action",
        "move": "Support the move action",
    }

    def __init__(
        self,
        toolbox: AgentToolbox,
        agent_memory: Memory,
        block_params: Optional[MobilityBlockParams] = None,
    ):
        super().__init__(
            toolbox=toolbox,
            agent_memory=agent_memory,
            block_params=block_params,
        )
        # initialize all blocks
        self.place_selection_block = PlaceSelectionBlock(
            toolbox, agent_memory, self.params.search_limit
        )
        self.move_block = MoveBlock(toolbox, agent_memory)
        self.trigger_time = 0  # Block invocation counter
        self.token_consumption = 0  # LLM token tracker

    async def forward(self, agent_context: DotDict) -> SocietyAgentMobilityBlockOutput:
        """Main entry point - delegates to sub-blocks"""
        self.trigger_time += 1
        context = agent_context | self.context
        next_place, NotFound = ("home", False)
        for _ in range(5):
            try:
                next_place, NotFound = await self.place_selection_block.forward(context)
                break
            except Exception as e:
                get_logger().warning(f"MobilityBlock.forward retry failed: {type(e).__name__}: {e}")
                continue

        # Debug print removed.
        result = await self.move_block.forward(next_place, context, NotFound=NotFound)
        return self.OutputType(**result)
