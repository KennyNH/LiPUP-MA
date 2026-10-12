"""Map tools for multimodal planning (relative-coordinate oriented)."""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from shapely.geometry import Polygon

try:
    from .landuse import LANDUSE_COLORS, LANDUSE_LABELS, MODIFIABLE_LANDUSE
except ImportError:
    from landuse import LANDUSE_COLORS, LANDUSE_LABELS, MODIFIABLE_LANDUSE

logger = logging.getLogger(__name__)


def _ensure_legacy_path(repo_root: Path) -> None:
    legacy_root = repo_root / "planning_agent"
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MapWorkspace:
    """Workspace wrapper for map conversion and visualization."""

    def __init__(self, repo_root: Path, output_dir: Path):
        self.repo_root = repo_root
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _ensure_legacy_path(repo_root)
        self._data_loader = None
        self._aoi_xy_positions: Dict[str, List[Dict[str, Any]]] = {}
        self._image_coord_extent: Dict[str, List[float]] = {}

    @property
    def data_loader(self):
        if self._data_loader is None:
            raise RuntimeError("DataLoader not initialized. call load_map() first.")
        return self._data_loader

    def ensure_geojson(self, pb_path: Path, geojson_cache: Optional[Path] = None) -> Path:
        geojson = geojson_cache or pb_path.with_suffix(".geojson")
        if geojson.exists():
            return geojson
        parser_mod = _load_module_from_file(
            "legacy_parse_map_geo",
            self.repo_root / "planning_agent" / "sim4pup_agent" / "data" / "parse_map_geo.py",
        )
        pb2geojson = getattr(parser_mod, "pb2geojson")
        pb2geojson(str(pb_path), str(geojson))
        if not geojson.exists():
            raise RuntimeError(f"Failed to convert pb to geojson: {pb_path}")
        return geojson

    def load_map(self, map_path: Path) -> None:
        data_loader_mod = _load_module_from_file(
            "legacy_data_loader",
            self.repo_root / "planning_agent" / "sim4pup_agent" / "tools" / "data_loader.py",
        )
        DataLoader = getattr(data_loader_mod, "DataLoader")

        suffix = map_path.suffix.lower()
        if suffix == ".pb":
            map_path = self.ensure_geojson(map_path)
            suffix = ".geojson"
        if suffix not in {".json", ".geojson"}:
            raise ValueError(f"Unsupported map file: {map_path}")
        data_format = "geojson" if suffix == ".geojson" else "json"
        loader = DataLoader(str(map_path), data_format=data_format)
        loader.load_gis_data()
        self._data_loader = loader
        if suffix == ".geojson":
            self._load_xy_positions_from_geojson(map_path)

    def _load_xy_positions_from_geojson(self, geojson_path: Path) -> None:
        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            feats = data.get("features") or []
            mapping: Dict[str, List[Dict[str, Any]]] = {}
            for ft in feats:
                props = ft.get("properties") or {}
                if props.get("feature_type") != "aoi":
                    continue
                aoi_id = str(ft.get("id"))
                positions = props.get("positions")
                if isinstance(positions, list) and len(positions) >= 3:
                    mapping[aoi_id] = positions
            self._aoi_xy_positions = mapping
        except Exception as e:
            logger.warning("Failed to read positions from geojson: %s", e)
            self._aoi_xy_positions = {}

    def map_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        gdf = self.data_loader.get_aois()
        if len(gdf) == 0:
            return None
        return tuple(gdf.total_bounds)

    def _to_positions_xy_gdf(self, gdf):
        g2 = gdf.copy()
        changed = 0
        for idx, row in g2.iterrows():
            aid = str(row["aoi_id"])
            pos = self._aoi_xy_positions.get(aid)
            if not pos:
                continue
            coords = []
            for p in pos:
                try:
                    coords.append((float(p["x"]), float(p["y"])))
                except Exception:
                    pass
            if len(coords) >= 3:
                try:
                    g2.at[idx, "geometry"] = Polygon(coords)
                    changed += 1
                except Exception:
                    pass
        if changed > 0:
            # These geometries are rewritten into local XY coordinates, so the
            # inherited geographic CRS metadata is no longer valid.
            g2.crs = None
        return g2

    def render_map(
        self,
        output_name: str,
        visible_aois: Optional[Iterable[str]] = None,
        focus_aois: Optional[Iterable[str]] = None,
        highlight_aois: Optional[Iterable[str]] = None,
        show_network: bool = False,
        proposed_overrides: Optional[Dict[str, str]] = None,
        annotate_modifiable: bool = True,
        label_aois: Optional[Iterable[str]] = None,
        label_bbox_rel: Optional[List[float]] = None,
        view_bbox_rel: Optional[List[float]] = None,
        use_relative_coords: bool = True,
        base_image_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (22, 15),
        dpi: int = 170,
    ) -> Dict[str, Any]:
        try:
            import matplotlib.image as mpimg
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
        except Exception as e:
            return {"error": f"matplotlib not available: {e}"}

        out_path = self.output_dir / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf = self.data_loader.get_aois().copy()
        if len(gdf) == 0:
            return {"error": "No AOI geometry available"}

        gdf["aoi_id"] = gdf["aoi_id"].astype(str)
        gdf["landuse_type"] = gdf["landuse_type"].astype(str)
        if visible_aois:
            keep = {str(x) for x in visible_aois}
            gdf = gdf[gdf["aoi_id"].isin(keep)]
            if len(gdf) == 0:
                return {"error": "No AOIs left after visible_aois filter"}
        if proposed_overrides:
            for aoi_id, target in proposed_overrides.items():
                mask = gdf["aoi_id"] == str(aoi_id)
                if mask.any():
                    gdf.loc[mask, "landuse_type"] = str(target)

        plot_gdf = gdf
        source_bounds = self.map_bounds()
        coord_extent: Optional[List[float]] = None
        if use_relative_coords:
            plot_gdf = self._to_positions_xy_gdf(gdf)

        plot_gdf["plot_color"] = plot_gdf["landuse_type"].map(lambda x: LANDUSE_COLORS.get(x, LANDUSE_COLORS["unknown"]))
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        if base_image_path and base_image_path.exists():
            img = mpimg.imread(str(base_image_path))
            if use_relative_coords:
                bext = self._image_coord_extent.get(str(base_image_path.resolve()))
                if not bext:
                    sidecar = Path(str(base_image_path) + ".coord.json")
                    if sidecar.exists():
                        try:
                            with open(sidecar, "r", encoding="utf-8") as f:
                                bext = (json.load(f) or {}).get("coord_extent")
                        except Exception:
                            bext = None
                if bext and len(bext) == 4:
                    ax.imshow(img, extent=(bext[0], bext[2], bext[1], bext[3]), zorder=0, alpha=0.95)
                else:
                    xmin, ymin, xmax, ymax = plot_gdf.total_bounds
                    ax.imshow(img, extent=(xmin, xmax, ymin, ymax), zorder=0, alpha=0.95)
            elif source_bounds:
                minx, miny, maxx, maxy = source_bounds
                ax.imshow(img, extent=(minx, maxx, miny, maxy), zorder=0, alpha=0.95)

        plot_gdf.plot(ax=ax, color=plot_gdf["plot_color"], edgecolor="#212121", linewidth=1.3, alpha=0.88, zorder=2)

        focus_ids = {str(x) for x in (focus_aois or [])}
        if focus_ids:
            fg = plot_gdf[plot_gdf["aoi_id"].isin(focus_ids)]
            if len(fg) > 0:
                fg.plot(ax=ax, color="none", edgecolor="#FFB300", linewidth=3.4, zorder=4)

        highlight_ids = {str(x) for x in (highlight_aois or [])}
        if highlight_ids:
            hg = plot_gdf[plot_gdf["aoi_id"].isin(highlight_ids)]
            if len(hg) > 0:
                hg.plot(ax=ax, facecolor="#FFF176", edgecolor="#E65100", linewidth=4.0, alpha=0.58, zorder=5)

        if show_network:
            try:
                G = self.data_loader.get_network_graph("walking")
                if G is not None:
                    if use_relative_coords:
                        pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True) if "x" in d and "y" in d}
                    else:
                        pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True) if "x" in d and "y" in d}
                    for u, v in G.edges():
                        if u in pos and v in pos:
                            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#616161", linewidth=0.35, alpha=0.30, zorder=1)
            except Exception as e:
                logger.warning("Failed to draw network: %s", e)

        labels = set(str(x) for x in (label_aois or []))
        if annotate_modifiable:
            labels.update(set(plot_gdf[plot_gdf["landuse_type"].isin(MODIFIABLE_LANDUSE)]["aoi_id"].tolist()))

        if label_bbox_rel and len(label_bbox_rel) == 4:
            x1, y1, x2, y2 = [float(v) for v in label_bbox_rel]
            if not use_relative_coords:
                logger.warning("label_bbox_rel only applies in relative coords mode")
            else:
                in_bbox = []
                for _, row in plot_gdf.iterrows():
                    c = row.geometry.centroid
                    if x1 <= c.x <= x2 and y1 <= c.y <= y2:
                        in_bbox.append(str(row["aoi_id"]))
                labels.update(in_bbox)

        for _, row in plot_gdf[plot_gdf["aoi_id"].isin(labels)].iterrows():
            c = row.geometry.centroid
            label_text = str(row["aoi_id"])
            # Keep readable and centered.
            try:
                label_text = str(int(str(row["aoi_id"])) % 1000)
            except Exception:
                pass
            is_hi = str(row["aoi_id"]) in highlight_ids
            ax.annotate(
                label_text,
                xy=(c.x, c.y),
                xytext=(0, 0),
                textcoords="offset points",
                fontsize=14 if not is_hi else 16,
                fontweight="bold",
                ha="center",
                va="center",
                color="#111111",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": "#FFF9C4" if is_hi else "#FFFFFF",
                    "edgecolor": "#E65100" if is_hi else "#111111",
                    "linewidth": 1.7 if is_hi else 1.0,
                    "alpha": 0.9,
                },
                zorder=20,
            )

        legend_items = []
        for lu in sorted(plot_gdf["landuse_type"].unique()):
            count = int((plot_gdf["landuse_type"] == lu).sum())
            legend_items.append((lu, count))
        handles = [
            mpatches.Patch(
                facecolor=LANDUSE_COLORS.get(lu, LANDUSE_COLORS["unknown"]),
                edgecolor="black",
                label=f"{LANDUSE_LABELS.get(lu, lu)} ({count})",
            )
            for lu, count in legend_items
        ]
        if highlight_ids:
            handles.append(
                mpatches.Patch(
                    facecolor="#FFF176",
                    edgecolor="#E65100",
                    linewidth=2,
                    label=f"Modified / Highlighted ({len(highlight_ids)})",
                )
            )
        if handles:
            # Place legend outside the plotting area to avoid covering AOI polygons.
            ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                fontsize=13,
                title="Land-use Legend",
                title_fontsize=14,
                borderpad=0.6,
                handlelength=1.25,
                labelspacing=0.45,
                framealpha=0.96,
            )

        if use_relative_coords:
            if view_bbox_rel and len(view_bbox_rel) == 4:
                x1, y1, x2, y2 = [float(v) for v in view_bbox_rel]
                xmin, xmax = sorted([x1, x2])
                ymin, ymax = sorted([y1, y2])
            else:
                xmin, ymin, xmax, ymax = plot_gdf.total_bounds
            mx = max((xmax - xmin) * 0.012, 0.4)
            my = max((ymax - ymin) * 0.012, 0.4)
            ax.set_xlim(xmin - mx, xmax + mx)
            ax.set_ylim(ymin - my, ymax + my)
            ax.set_xlabel("X (positions coordinate)", fontsize=15)
            ax.set_ylabel("Y (positions coordinate)", fontsize=15)
            coord_extent = [float(xmin - mx), float(ymin - my), float(xmax + mx), float(ymax + my)]
        elif source_bounds:
            minx, miny, maxx, maxy = source_bounds
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_xlabel("Longitude", fontsize=15)
            ax.set_ylabel("Latitude", fontsize=15)
            coord_extent = [float(minx), float(miny), float(maxx), float(maxy)]
        ax.tick_params(axis="both", labelsize=13)
        ax.set_title("Planning Map", fontsize=18, pad=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.8)
        # Reserve right margin for external legend so map content stays unobstructed.
        fig.subplots_adjust(left=0.03, right=0.82, top=0.962, bottom=0.05)
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        if coord_extent:
            self._image_coord_extent[str(out_path.resolve())] = coord_extent
            sidecar = Path(str(out_path) + ".coord.json")
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump({"coord_extent": coord_extent, "coord_mode": "positions_xy" if use_relative_coords else "epsg4326"}, f, ensure_ascii=False, indent=2)

        return {
            "output_path": str(out_path.resolve()),
            "landuse_legend": [{"landuse": lu, "label": LANDUSE_LABELS.get(lu, lu), "count": cnt} for lu, cnt in legend_items],
            "coord_mode": "positions_xy" if use_relative_coords else "geographic_epsg4326",
            "coord_extent": coord_extent,
            "success": True,
        }

    def image_zoom(
        self,
        image_path: Path,
        bbox_2d: List[float],
        output_name: str,
        magnify: float = 2.0,
        min_side_px: int = 256,
        source_coord_extent: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Zoom selected region by coordinate bbox + enlargement, and display coordinate axes."""
        try:
            from PIL import Image
            import matplotlib.pyplot as plt
        except Exception as e:
            return {"error": f"PIL not available: {e}"}
        if len(bbox_2d) != 4:
            return {"error": "bbox_2d must be [x1,y1,x2,y2]"}

        img = Image.open(str(image_path))
        w, h = img.size
        ext = source_coord_extent
        if not ext:
            ext = self._image_coord_extent.get(str(image_path.resolve()))
        if not ext:
            sidecar = Path(str(image_path) + ".coord.json")
            if sidecar.exists():
                try:
                    with open(sidecar, "r", encoding="utf-8") as f:
                        ext = (json.load(f) or {}).get("coord_extent")
                except Exception:
                    ext = None
        if not ext or len(ext) != 4:
            # fallback for old behavior: treat bbox_2d as normalized 0..1000
            xmin, ymin, xmax, ymax = 0.0, 0.0, 1000.0, 1000.0
        else:
            xmin, ymin, xmax, ymax = [float(v) for v in ext]

        x1, y1, x2, y2 = [float(v) for v in bbox_2d]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        dx = max(xmax - xmin, 1e-12)
        dy = max(ymax - ymin, 1e-12)
        px1 = int(max(0, min(w - 1, (x1 - xmin) / dx * w)))
        py1 = int(max(0, min(h - 1, (ymax - y2) / dy * h)))
        px2 = int(max(px1 + 1, min(w, (x2 - xmin) / dx * w)))
        py2 = int(max(py1 + 1, min(h, (ymax - y1) / dy * h)))

        crop = img.crop((px1, py1, px2, py2))
        cw, ch = crop.size
        base_scale = max(min_side_px / max(min(cw, ch), 1), 1.0)
        scale = max(float(magnify), 1.0) * base_scale
        nw = int(math.ceil(cw * scale))
        nh = int(math.ceil(ch * scale))
        zoomed = crop.resize((nw, nh), resample=Image.BICUBIC)

        output_path = self.output_dir / output_name
        # Draw zoomed result with coordinate axes in the same coordinate system.
        fig_w = max(6.0, nw / 220.0)
        fig_h = max(5.0, nh / 220.0)
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=140)
        ax.imshow(zoomed, extent=(x1, x2, y1, y2), origin="upper")
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        ax.set_xlabel("X (positions coordinate)")
        ax.set_ylabel("Y (positions coordinate)")
        ax.set_title("Zoomed Region")
        ax.grid(alpha=0.2, linestyle="--")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        out_ext = [x1, y1, x2, y2]
        self._image_coord_extent[str(output_path.resolve())] = out_ext
        with open(Path(str(output_path) + ".coord.json"), "w", encoding="utf-8") as f:
            json.dump({"coord_extent": out_ext, "coord_mode": "positions_xy"}, f, ensure_ascii=False, indent=2)
        return {
            "output_path": str(output_path.resolve()),
            "pixel_bbox": [px1, py1, px2, py2],
            "scale": round(scale, 3),
            "coord_extent": out_ext,
            "success": True,
        }


def load_map_snapshot_from_pb(repo_root: Path, pb_path: Path, output_geojson: Path) -> Dict[str, Any]:
    ws = MapWorkspace(repo_root=repo_root, output_dir=output_geojson.parent)
    geojson = ws.ensure_geojson(pb_path=pb_path, geojson_cache=output_geojson)
    with open(geojson, "r", encoding="utf-8") as f:
        data = json.load(f)
    feats = data.get("features") or []
    aois = [x for x in feats if (x.get("properties") or {}).get("feature_type") == "aoi"]
    return {
        "geojson_path": str(geojson),
        "feature_count": len(feats),
        "aoi_feature_count": len(aois),
        "bbox": data.get("bbox"),
        "header": data.get("header"),
    }
