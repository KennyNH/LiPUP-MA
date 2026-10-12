"""Map edits and protobuf serialization used by the simulation/planning loop."""
import pickle
from google.protobuf.json_format import ParseDict
from pycityproto.city.map.v2.map_pb2 import Map


def modify_aois(modify_dict, path, new_path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    aois = {aoi["id"]: aoi for aoi in data["aois"]}
    for aoi_id, land_use in modify_dict.items():
        aois[int(aoi_id)]["urban_land_use"] = land_use
    with open(new_path, "wb") as f:
        pickle.dump(data, f)


def dict2pb_path(path_dict, path_pb):
    with open(path_dict, "rb") as f:
        data = pickle.load(f)
    message = ParseDict(data, Map(), ignore_unknown_fields=True)
    with open(path_pb, "wb") as f:
        f.write(message.SerializeToString())
