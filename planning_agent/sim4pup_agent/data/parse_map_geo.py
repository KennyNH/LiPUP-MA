#!/usr/bin/env python3
"""
完整解析 AgentSociety 地图 .pb 文件，输出为标准 GeoJSON (RFC 7946)。

- AOI 转为 Feature，geometry 为 Polygon（或退化为 Point），经纬度坐标。
- POI 转为 Feature，geometry 为 Point，经纬度坐标。
- 输出为单个 FeatureCollection，可选带 bbox、metadata、header。
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


try:
    from agentsociety.environment.mapdata import MapData, MapConfig
    from agentsociety.s3 import S3Config
    from shapely.geometry import Point, Polygon
    print("✅ AgentSociety模块导入成功")
except ImportError as e:
    print(f"❌ 无法导入AgentSociety: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# GeoJSON 标准几何 (RFC 7946)
# ---------------------------------------------------------------------------

def shapely_to_geojson_geometry(geom) -> Optional[Dict[str, Any]]:
    """
    将 Shapely 几何（经纬度）转为 GeoJSON Geometry 对象。
    坐标顺序为 [longitude, latitude]。
    """
    if geom is None:
        return None
    if isinstance(geom, Point):
        return {
            "type": "Point",
            "coordinates": [geom.x, geom.y]
        }
    if isinstance(geom, Polygon):
        # GeoJSON Polygon: coordinates = [ exterior_ring, hole1, ... ]
        # exterior_ring = [ [lng, lat], ... ]，首尾需闭合
        exterior = [[float(c[0]), float(c[1])] for c in geom.exterior.coords]
        coords = [exterior]
        for interior in geom.interiors:
            coords.append([[float(c[0]), float(c[1])] for c in interior.coords])
        return {
            "type": "Polygon",
            "coordinates": coords
        }
    return None


def properties_from_aoi(aoi: Dict[str, Any]) -> Dict[str, Any]:
    """从 AOI 字典提取 properties（排除几何字段，保证可 JSON 序列化）。"""
    skip = {"shapely_xy", "shapely_lnglat"}
    return {k: v for k, v in aoi.items() if k not in skip}


def properties_from_poi(poi: Dict[str, Any]) -> Dict[str, Any]:
    """从 POI 字典提取 properties（排除几何字段）。"""
    skip = {"shapely_xy", "shapely_lnglat"}
    return {k: v for k, v in poi.items() if k not in skip}


def compute_bbox_from_features(features: List[Dict]) -> Optional[List[float]]:
    """根据所有 Feature 的 geometry 计算 FeatureCollection 的 bbox [minLng, minLat, maxLng, maxLat]。"""
    min_lng, min_lat, max_lng, max_lat = None, None, None, None

    def update_point(lng: float, lat: float):
        nonlocal min_lng, min_lat, max_lng, max_lat
        if min_lng is None:
            min_lng, min_lat, max_lng, max_lat = lng, lat, lng, lat
        else:
            min_lng, max_lng = min(min_lng, lng), max(max_lng, lng)
            min_lat, max_lat = min(min_lat, lat), max(max_lat, lat)

    def walk_coords(coords):
        if not coords:
            return
        if isinstance(coords[0], (int, float)):
            update_point(float(coords[0]), float(coords[1]))
        else:
            for ring in coords:
                for pt in ring:
                    update_point(float(pt[0]), float(pt[1]))

    for f in features:
        geom = f.get("geometry")
        if not geom:
            continue
        walk_coords(geom.get("coordinates"))

    if min_lng is None:
        return None
    return [min_lng, min_lat, max_lng, max_lat]


def parse_complete_map_data(pb_file_path: str, output_file: str) -> bool:
    """
    解析 .pb 地图并输出为标准 GeoJSON FeatureCollection。

    Args:
        pb_file_path: .pb 文件路径
        output_file: 输出 GeoJSON 文件路径
    """
    print("📁 开始解析地图数据（GeoJSON 输出）...")
    print(f"   输入: {pb_file_path}")
    print(f"   输出: {output_file}")

    try:
        map_config = MapConfig(file_path=pb_file_path)
        s3_config = S3Config(enabled=False)
        map_data = MapData(map_config, s3_config)

        print("✅ 地图数据加载成功")

        features: List[Dict[str, Any]] = []

        # 1. AOI → Feature (Polygon 或 Point)
        print("\n🏢 解析 AOI...")
        for aoi_id, aoi in map_data.aois.items():
            geom = aoi.get("shapely_lnglat")
            geojson_geom = shapely_to_geojson_geometry(geom)
            if geojson_geom is None:
                continue
            feature = {
                "type": "Feature",
                "geometry": geojson_geom,
                "properties": properties_from_aoi(aoi)
            }
            feature["properties"]["feature_type"] = "aoi"
            feature["id"] = aoi_id
            features.append(feature)

        aoi_count = len(features)
        print(f"   ✅ 解析了 {aoi_count} 个 AOI")

        # 2. POI → Feature (Point)
        print("\n📍 解析 POI...")
        for poi_id, poi in map_data.pois.items():
            geom = poi.get("shapely_lnglat")
            geojson_geom = shapely_to_geojson_geometry(geom)
            if geojson_geom is None:
                continue
            feature = {
                "type": "Feature",
                "geometry": geojson_geom,
                "properties": properties_from_poi(poi)
            }
            feature["properties"]["feature_type"] = "poi"
            feature["id"] = poi_id
            features.append(feature)

        poi_count = len(features) - aoi_count
        print(f"   ✅ 解析了 {poi_count} 个 POI")

        # 3. 组装 GeoJSON FeatureCollection
        bbox = compute_bbox_from_features(features)
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        if bbox is not None:
            geojson["bbox"] = bbox
        # 非标准键，便于保留元数据（RFC 7946 允许 foreign members）
        geojson["metadata"] = {
            "source_file": pb_file_path,
            "description": "AgentSociety 地图导出为 GeoJSON，含 AOI（面/点）与 POI（点）"
        }
        geojson["header"] = dict(map_data.header)

        # 4. 写入文件
        print("\n💾 写入 GeoJSON...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print("\n🎉 解析完成")
        print(f"📄 输出: {output_file}")
        print(f"📊 大小: {size_mb:.2f} MB")
        print(f"   AOI: {aoi_count}, POI: {poi_count}, features: {len(features)}")

        return True

    except Exception as e:
        print(f"❌ 解析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_completeness(pb_file_path: str, geojson_file_path: str) -> bool:
    """验证 GeoJSON 中 feature 数量与 .pb 一致。"""
    print("\n🔍 验证数据完整性...")

    try:
        map_config = MapConfig(file_path=pb_file_path)
        s3_config = S3Config(enabled=False)
        map_data = MapData(map_config, s3_config)

        with open(geojson_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("type") != "FeatureCollection":
            print("   ⚠️ 根类型不是 FeatureCollection")
            return False

        features = data.get("features", [])
        aoi_in_geojson = sum(1 for f in features if f.get("properties", {}).get("feature_type") == "aoi")
        poi_in_geojson = sum(1 for f in features if f.get("properties", {}).get("feature_type") == "poi")

        print(f"   AOI: 原始={len(map_data.aois)}, GeoJSON={aoi_in_geojson} {'✓' if len(map_data.aois) == aoi_in_geojson else '✗'}")
        print(f"   POI: 原始={len(map_data.pois)}, GeoJSON={poi_in_geojson} {'✓' if len(map_data.pois) == poi_in_geojson else '✗'}")
        print("✅ 验证完成")
        return True

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    base_dir = Path(__file__).parent
    pb_file = str(base_dir.parent / "data" / "HKWJ.pb")
    output_file = str(base_dir / "HKWJ.geojson")

    if not os.path.exists(pb_file):
        print(f"❌ PB 文件不存在: {pb_file}")
        return

    if parse_complete_map_data(pb_file, output_file):
        verify_completeness(pb_file, output_file)


def pb2geojson(pb_path,geojson_path):
    base_dir = Path(__file__).parent

    if not os.path.exists(pb_path):
        print(f"❌ PB 文件不存在: {pb_path}")
        return

    if parse_complete_map_data(pb_path, geojson_path):
        verify_completeness(pb_path, geojson_path)


