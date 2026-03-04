from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def distance_xy(a_xyz: list[float], b_xyz: list[float]) -> float:
    return math.hypot(a_xyz[0] - b_xyz[0], a_xyz[1] - b_xyz[1])


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_database(nuscenes_meta_dir: Path, out_db_path: Path) -> int:
    sample = load_json(nuscenes_meta_dir / "sample.json")
    sample_data = load_json(nuscenes_meta_dir / "sample_data.json")
    sample_annotation = load_json(nuscenes_meta_dir / "sample_annotation.json")
    ego_pose = load_json(nuscenes_meta_dir / "ego_pose.json")
    calibrated_sensor = load_json(nuscenes_meta_dir / "calibrated_sensor.json")
    sensor = load_json(nuscenes_meta_dir / "sensor.json")
    instance = load_json(nuscenes_meta_dir / "instance.json")
    category = load_json(nuscenes_meta_dir / "category.json")
    visibility = load_json(nuscenes_meta_dir / "visibility.json")

    category_by_token = {row["token"]: row["name"] for row in category}
    instance_category = {row["token"]: category_by_token[row["category_token"]] for row in instance}
    pose_by_token = {row["token"]: row for row in ego_pose}
    calib_by_token = {row["token"]: row for row in calibrated_sensor}
    sensor_by_token = {row["token"]: row for row in sensor}
    visibility_score = {
        row["token"]: {"v0-40": 0.20, "v40-60": 0.50, "v60-80": 0.70, "v80-100": 0.90}[row["level"]]
        for row in visibility
    }

    keyframe_by_sample_and_channel: dict[tuple[str, str], dict] = {}
    for row in sample_data:
        if not row["is_key_frame"]:
            continue
        sensor_token = calib_by_token[row["calibrated_sensor_token"]]["sensor_token"]
        channel = sensor_by_token[sensor_token]["channel"]
        keyframe_by_sample_and_channel[(row["sample_token"], channel)] = row

    ann_by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in sample_annotation:
        ann_by_sample[row["sample_token"]].append(row)

    rows = []
    sorted_samples = sorted(sample, key=lambda row: row["timestamp"])
    for idx, row in enumerate(sorted_samples):
        sample_token = row["token"]
        lidar_kf = keyframe_by_sample_and_channel.get((sample_token, "LIDAR_TOP"))
        if lidar_kf is None:
            continue

        ego_translation = pose_by_token[lidar_kf["ego_pose_token"]]["translation"]
        annotations = ann_by_sample.get(sample_token, [])

        near_objects = []
        near_pedestrians = []
        for ann in annotations:
            cat_name = instance_category.get(ann["instance_token"], "")
            dist = distance_xy(ann["translation"], ego_translation)
            if dist <= 40.0:
                near_objects.append((ann, cat_name, dist))
            if cat_name.startswith("human.pedestrian") and dist <= 20.0 and ann["num_lidar_pts"] > 0:
                near_pedestrians.append(ann)

        # Binary task for benchmarking: pedestrian-presence risk at close range.
        label = 1 if near_pedestrians else 0

        if near_objects:
            vis_avg = mean(visibility_score.get(ann["visibility_token"], 0.5) for ann, _, _ in near_objects)
            object_factor = clamp(len(near_objects) / 12.0)
            lidar_pts_avg = mean(float(ann["num_lidar_pts"]) for ann, _, _ in near_objects)
            radar_pts_avg = mean(float(ann["num_radar_pts"]) for ann, _, _ in near_objects)

            camera_score = clamp(0.25 + 0.50 * vis_avg + 0.25 * object_factor)
            lidar_score = clamp(0.20 + 0.80 * clamp(lidar_pts_avg / 20.0))
            radar_score = clamp(0.15 + 0.85 * clamp(radar_pts_avg / 8.0))
        else:
            camera_score = 0.15
            lidar_score = 0.10
            radar_score = 0.10

        camera_uncertainty = clamp(1.0 - camera_score, 0.05, 0.95)
        lidar_uncertainty = clamp(1.0 - lidar_score, 0.05, 0.95)
        radar_uncertainty = clamp(1.0 - radar_score, 0.05, 0.95)

        rows.append(
            (
                idx,
                int(row["timestamp"]),
                label,
                camera_score,
                camera_uncertainty,
                lidar_score,
                lidar_uncertainty,
                radar_score,
                radar_uncertainty,
            )
        )

    out_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(out_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                sample_id INTEGER PRIMARY KEY,
                timestamp INTEGER,
                label INTEGER NOT NULL,
                camera_score REAL NOT NULL,
                camera_uncertainty REAL NOT NULL,
                lidar_score REAL NOT NULL,
                lidar_uncertainty REAL NOT NULL,
                radar_score REAL NOT NULL,
                radar_uncertainty REAL NOT NULL
            )
            """
        )
        cur.execute("DELETE FROM samples")
        cur.executemany(
            """
            INSERT INTO samples (
                sample_id,
                timestamp,
                label,
                camera_score,
                camera_uncertainty,
                lidar_score,
                lidar_uncertainty,
                radar_score,
                radar_uncertainty
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fusionbench SQLite DB from nuScenes metadata.")
    parser.add_argument(
        "--nuscenes-meta-dir",
        default="data/v1.0-mini",
        help="Directory containing nuScenes v1.0-mini JSON metadata files.",
    )
    parser.add_argument(
        "--out-db",
        default="data/nuscenes_mini_samples.db",
        help="Output SQLite DB path with fusionbench samples table.",
    )
    args = parser.parse_args()

    count = build_database(Path(args.nuscenes_meta_dir), Path(args.out_db))
    print(f"Wrote {count} rows to {args.out_db}")


if __name__ == "__main__":
    main()
