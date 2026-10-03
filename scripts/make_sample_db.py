"""Create data/sample.db: a SYNTHETIC missions database for the Analyst's Text-to-SQL tool.
Seeded, so the file is reproducible. None of this is real operational data."""
from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERRAINS = ["urban", "forest", "desert", "mountain", "coastal"]
ALGOS = ["boids", "pso", "cbba", "leader_follower"]
ANOMALIES = ["gps_loss", "link_drop", "motor_overcurrent", "low_battery"]


def build(path: Path, n_missions: int = 80, seed: int = 42) -> None:
    rng = random.Random(seed)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE missions (
          id INTEGER PRIMARY KEY, name TEXT, terrain TEXT, swarm_algorithm TEXT,
          uav_count INTEGER, duration_min REAL, battery_used_pct REAL, area_km2 REAL,
          success INTEGER, flown_on TEXT
        );
        CREATE TABLE anomalies (
          id INTEGER PRIMARY KEY, mission_id INTEGER REFERENCES missions(id),
          type TEXT, severity INTEGER
        );
        """
    )
    for i in range(1, n_missions + 1):
        terrain, algo = rng.choice(TERRAINS), rng.choice(ALGOS)
        uavs = rng.randint(3, 20)
        duration = round(rng.uniform(15, 120), 1)
        battery = round(min(98.0, duration * rng.uniform(0.5, 0.95) + (8 if terrain == "mountain" else 0)), 1)
        area = round(uavs * rng.uniform(0.4, 1.6), 2)
        p_ok = 0.9 - (0.08 if terrain in ("mountain", "urban") else 0) - (0.1 if algo == "leader_follower" else 0)
        ok = 1 if rng.random() < p_ok and battery < 95 else 0
        day = f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        conn.execute(
            "INSERT INTO missions VALUES (?,?,?,?,?,?,?,?,?,?)",
            (i, f"Mission-{i:03d}", terrain, algo, uavs, duration, battery, area, ok, day),
        )
    aid = 1
    for i in range(1, n_missions + 1):
        for _ in range(rng.choice([0, 0, 1, 1, 2])):
            conn.execute(
                "INSERT INTO anomalies VALUES (?,?,?,?)",
                (aid, i, rng.choice(ANOMALIES), rng.randint(1, 3)),
            )
            aid += 1
    conn.commit()
    conn.close()


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "sample.db"
    build(out)
    print(f"wrote {out}")
