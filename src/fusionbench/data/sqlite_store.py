from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

from fusionbench.core.types import Sample, SensorReading
from fusionbench.data.synthetic import generate_synthetic_samples


def create_demo_database(db_path: str, n_samples: int = 1500, seed: int = 42) -> str:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
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

        samples = generate_synthetic_samples(n_samples=n_samples, seed=seed)
        rows = [
            (
                sample.sample_id,
                sample.timestamp,
                sample.label,
                sample.sensors["camera"].score,
                sample.sensors["camera"].uncertainty,
                sample.sensors["lidar"].score,
                sample.sensors["lidar"].uncertainty,
                sample.sensors["radar"].score,
                sample.sensors["radar"].uncertainty,
            )
            for sample in samples
        ]

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

    return str(path)


def load_samples_from_database(db_path: str, table: str = "samples", limit: int | None = None) -> List[Sample]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        query = (
            f"SELECT sample_id, timestamp, label, camera_score, camera_uncertainty, "
            f"lidar_score, lidar_uncertainty, radar_score, radar_uncertainty FROM {table} ORDER BY sample_id"
        )
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        cur.execute(query)
        rows = cur.fetchall()
    finally:
        conn.close()

    samples: List[Sample] = []
    for row in rows:
        (
            sample_id,
            timestamp,
            label,
            camera_score,
            camera_uncertainty,
            lidar_score,
            lidar_uncertainty,
            radar_score,
            radar_uncertainty,
        ) = row
        samples.append(
            Sample(
                sample_id=int(sample_id),
                timestamp=int(timestamp) if timestamp is not None else None,
                label=int(label),
                sensors={
                    "camera": SensorReading(float(camera_score), float(camera_uncertainty)),
                    "lidar": SensorReading(float(lidar_score), float(lidar_uncertainty)),
                    "radar": SensorReading(float(radar_score), float(radar_uncertainty)),
                },
            )
        )

    return samples

