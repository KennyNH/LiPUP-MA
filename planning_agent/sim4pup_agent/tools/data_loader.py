#!/usr/bin/env python3
"""
Data loader for GIS AOI (and optional POI/buildings). JSON or GeoJSON.
Uses real geometry only: no synthetic rectangles.
- GeoJSON (.geojson): FeatureCollection with WGS84 coordinates, best for visualization.
- JSON (.json): aois dict or geometry.coordinates / shapely_xy / positions.
Optional: OSM/buildings from buildings_path or map JSON key "buildings".
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import geopandas as gpd
import networkx as nx
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


def parse_aoi_geometry(aoi_data: dict) -> Optional[Polygon]:
    """
    Parse AOI geometry from one of the supported formats.
    Supports: geometry.coordinates (MOSS/GeoJSON), shapely_xy, positions.
    Returns None if no valid geometry (no synthetic polygons).
    """
    # 1. MOSS/GeoJSON: geometry.coordinates
    geom = aoi_data.get('geometry')
    if geom and isinstance(geom, dict):
        coords_raw = geom.get('coordinates')
        if isinstance(coords_raw, list) and len(coords_raw) > 0:
            ring = coords_raw[0]
            if isinstance(ring, list) and len(ring) >= 3:
                try:
                    return Polygon([(float(p[0]), float(p[1])) for p in ring])
                except (TypeError, IndexError):
                    pass

    # 2. shapely_xy (AgentSociety / legacy)
    shapely_data = aoi_data.get('shapely_xy')
    if shapely_data:
        if isinstance(shapely_data, dict):
            if shapely_data.get('type') == 'Polygon' and 'coordinates' in shapely_data:
                coords = shapely_data['coordinates']
                if isinstance(coords, list) and len(coords) > 0 and len(coords[0]) >= 3:
                    return Polygon(coords[0])
        elif isinstance(shapely_data, list) and len(shapely_data) >= 3:
            return Polygon(shapely_data)

    # 3. positions (list of {x,y} or {lon,lat})
    positions = aoi_data.get('positions')
    if isinstance(positions, list) and len(positions) >= 3:
        try:
            coords = [
                (float(p.get('x', p.get('lon', 0))), float(p.get('y', p.get('lat', 0))))
                for p in positions
            ]
            return Polygon(coords)
        except (TypeError, ValueError):
            pass

    return None


class DataLoader:
    """
    GIS data loader: loads AOI (and optional POI, buildings) from JSON.
    Uses real geometry only; AOIs without geometry are skipped (no rectangular synthesis).
    For OSM-style maps, provide buildings_json or map JSON with "buildings" key.
    """

    def __init__(self, data_path: str, data_format: str = 'auto', buildings_path: Optional[str] = None):
        data_path = Path(data_path)
        if not data_path.exists():
            raise ValueError(f"Data file does not exist: {data_path}")
        logger.info("Initializing DataLoader: %s", data_path)

        self.data_path = data_path
        if data_format == 'auto':
            suf = data_path.suffix.lower()
            if suf == '.json':
                data_format = 'json'
            elif suf == '.geojson':
                data_format = 'geojson'
            else:
                raise ValueError("Use a .json or .geojson file.")
        elif data_format not in ('json', 'geojson'):
            raise ValueError("data_format must be 'json', 'geojson', or 'auto'.")
        self.data_format = data_format
        self.buildings_path = Path(buildings_path) if buildings_path else None

        self.gdf_aois: Optional[gpd.GeoDataFrame] = None
        self.gdf_poi: Optional[gpd.GeoDataFrame] = None
        self.gdf_buildings: Optional[gpd.GeoDataFrame] = None
        self.G_walking: Optional[nx.MultiDiGraph] = None
        self.G_driving: Optional[nx.MultiDiGraph] = None
        self.map_dict: Optional[Dict] = None

    def load_gis_data(self) -> None:
        """Load AOI (and optional POI, buildings) from JSON or GeoJSON. Real geometry only."""
        logger.info("Loading GIS data...")
        try:
            if self.data_format == 'geojson':
                self._load_geojson_data()
            else:
                self._load_json_data()
            self._load_buildings_if_configured()
            logger.info("GIS data loaded successfully")
        except Exception as e:
            logger.exception("Failed to load GIS data")
            raise ValueError(f"Failed to load GIS data: {e}") from e

    def _load_json_data(self) -> None:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, dict):
            raise ValueError("JSON root must be an object")
        aois_data = raw_data.get('aois', raw_data)
        if not aois_data:
            aois_data = {}

        records: List[Dict[str, Any]] = []
        skipped_no_geom = 0

        for aoi_key, aoi_data in aois_data.items():
            try:
                aoi_id = str(aoi_data.get('id', aoi_data.get('aoi_id', aoi_key)))
                geom = parse_aoi_geometry(aoi_data)
                if geom is None:
                    skipped_no_geom += 1
                    continue
                record = {
                    'aoi_id': aoi_id,
                    'landuse_type': aoi_data.get('urban_land_use', 'unknown'),
                    'name': aoi_data.get('name', '') or f'AOI_{aoi_id}',
                    'area': aoi_data.get('area', geom.area),
                    'geometry': geom,
                }
                records.append(record)
            except Exception as e:
                logger.warning("Failed to parse AOI %s: %s", aoi_key, e)
                continue

        if skipped_no_geom:
            logger.info(
                "Skipped %d AOIs without geometry (use full map JSON or buildings_json for OSM layer).",
                skipped_no_geom,
            )

        if records:
            self.gdf_aois = gpd.GeoDataFrame(records, crs='EPSG:4326')
            invalid_geom = ~self.gdf_aois.geometry.is_valid
            if invalid_geom.any():
                logger.warning("Fixing %d invalid geometries", invalid_geom.sum())
                self.gdf_aois.loc[invalid_geom, 'geometry'] = (
                    self.gdf_aois.loc[invalid_geom, 'geometry'].buffer(0)
                )
            logger.info("Loaded %d AOIs with geometry", len(self.gdf_aois))
        else:
            # Empty AOI GeoDataFrame so get_aois() and visualizer can run (e.g. buildings-only map)
            self.gdf_aois = gpd.GeoDataFrame(
                columns=['aoi_id', 'landuse_type', 'name', 'area', 'geometry'],
                geometry='geometry',
                crs='EPSG:4326',
            )
            logger.info("No AOIs with geometry; AOI layer will be empty (use buildings_json for OSM).")

        if 'pois' in raw_data:
            self._load_poi_from_json(raw_data['pois'])
        if raw_data.get('buildings'):
            self._load_buildings_from_dict(raw_data['buildings'])
        self._create_map_dict()

    def _load_geojson_data(self) -> None:
        """Load AOI from GeoJSON FeatureCollection (WGS84). Best for visualization."""
        gdf = gpd.read_file(self.data_path)
        if gdf.crs is None:
            gdf.set_crs('EPSG:4326', inplace=True)
        else:
            gdf = gdf.to_crs('EPSG:4326')
        # Standardize to aoi_id, landuse_type, name, area (from GeoJSON properties)
        gdf['aoi_id'] = gdf['id'].astype(str) if 'id' in gdf.columns else gdf.index.astype(str)
        gdf['landuse_type'] = gdf['urban_land_use'].astype(str) if 'urban_land_use' in gdf.columns else 'unknown'
        if 'name' in gdf.columns:
            gdf['name'] = gdf['name'].fillna('').astype(str)
        else:
            gdf['name'] = ''
        gdf['area'] = gdf['area'] if 'area' in gdf.columns else gdf.geometry.area
        self.gdf_aois = gdf[['aoi_id', 'landuse_type', 'name', 'area', 'geometry']].copy()
        invalid_geom = ~self.gdf_aois.geometry.is_valid
        if invalid_geom.any():
            logger.warning("Fixing %d invalid geometries", invalid_geom.sum())
            self.gdf_aois.loc[invalid_geom, 'geometry'] = (
                self.gdf_aois.loc[invalid_geom, 'geometry'].buffer(0)
            )
        logger.info("Loaded %d AOIs from GeoJSON (WGS84)", len(self.gdf_aois))
        self._create_map_dict()

    def _load_poi_from_json(self, pois_data: dict) -> None:
        records = []
        for poi_key, poi_data in pois_data.items():
            try:
                geom = parse_aoi_geometry(poi_data)
                if geom is None:
                    continue
                records.append({
                    'poi_id': str(poi_data.get('id', poi_key)),
                    'category': poi_data.get('category', 'unknown'),
                    'name': poi_data.get('name', ''),
                    'geometry': geom,
                })
            except Exception:
                continue
        if records:
            self.gdf_poi = gpd.GeoDataFrame(records, crs='EPSG:4326')
            logger.info("Loaded %d POIs", len(self.gdf_poi))

    def _create_map_dict(self) -> None:
        self.map_dict = {}
        if self.gdf_aois is not None and len(self.gdf_aois) > 0:
            for _, row in self.gdf_aois.iterrows():
                self.map_dict[row['aoi_id']] = {
                    'geom': row['geometry'],
                    'use': row['landuse_type'],
                    'area': row['area'],
                    'name': row['name'],
                }

    def get_aois(self) -> gpd.GeoDataFrame:
        if self.gdf_aois is None:
            self.load_gis_data()
        if self.gdf_aois is None:
            raise ValueError("Failed to load AOI data")
        return self.gdf_aois

    def get_network_graph(self, network_type: str = 'walking') -> Optional[nx.MultiDiGraph]:
        if network_type not in ('walking', 'driving'):
            raise ValueError(f"Invalid network type: {network_type}")
        return self.G_walking if network_type == 'walking' else self.G_driving

    def get_buildings(self) -> Optional[gpd.GeoDataFrame]:
        """Return buildings GeoDataFrame if loaded (OSM/buildings GeoJSON)."""
        return self.gdf_buildings

    def _load_buildings_if_configured(self) -> None:
        if not self.buildings_path or not self.buildings_path.exists():
            return
        try:
            with open(self.buildings_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._load_buildings_from_dict(data)
            logger.info("Loaded buildings from %s: %d features", self.buildings_path, len(self.gdf_buildings))
        except Exception as e:
            logger.warning("Failed to load buildings from %s: %s", self.buildings_path, e)

    def _load_buildings_from_dict(self, data: Any) -> None:
        """Build gdf_buildings from GeoJSON-like dict (FeatureCollection or list of features)."""
        if not isinstance(data, (dict, list)):
            return
        features = []
        if isinstance(data, dict) and data.get('type') == 'FeatureCollection':
            features = data.get('features', [])
        elif isinstance(data, list):
            features = data
        if not features:
            return
        records = []
        for i, f in enumerate(features):
            if not isinstance(f, dict):
                continue
            geom_dict = f.get('geometry') if isinstance(f.get('geometry'), dict) else None
            if not geom_dict:
                continue
            try:
                geom = shape(geom_dict)
                if geom is None or geom.is_empty:
                    continue
                if isinstance(geom, BaseGeometry):
                    records.append({'geometry': geom, 'id': f.get('id', i)})
            except Exception:
                continue
        if records:
            self.gdf_buildings = gpd.GeoDataFrame(records, crs='EPSG:4326')
            invalid = ~self.gdf_buildings.geometry.is_valid
            if invalid.any():
                self.gdf_buildings.loc[invalid, 'geometry'] = (
                    self.gdf_buildings.loc[invalid, 'geometry'].buffer(0)
                )
            logger.info("Loaded %d building footprints", len(self.gdf_buildings))
