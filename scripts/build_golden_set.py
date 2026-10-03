"""Build data/eval/golden_set.jsonl: 30 document questions (hand-written, grounded in data/raw)
+ 20 database questions whose ground truth is computed from data/sample.db with SQL."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (question, ground_truth, keywords that must all appear in a correct answer, expected source file)
DOC_QS = [
    # swarm_coordination.md
    ("What three local rules does the Boids flocking model use?", "Separation, alignment and cohesion.", ["separation", "alignment", "cohesion"], "swarm_coordination.md"),
    ("Who introduced Particle Swarm Optimization and in what year?", "Kennedy and Eberhart, 1995.", ["kennedy", "eberhart", "1995"], "swarm_coordination.md"),
    ("What is CBBA and how does it allocate tasks?", "Consensus-Based Bundle Algorithm, a decentralized auction-based method.", ["consensus-based bundle", "auction"], "swarm_coordination.md"),
    ("What is the time complexity of the Hungarian algorithm for assignment?", "O(n^3).", ["n^3"], "swarm_coordination.md"),
    ("What is the main weakness of leader-follower swarm control?", "The leader is a single point of failure.", ["single point of failure"], "swarm_coordination.md"),
    ("Who proposed Ant Colony Optimization?", "Marco Dorigo.", ["dorigo"], "swarm_coordination.md"),
    # uav_communications.md
    ("Who created MAVLink and when was it first released?", "Lorenz Meier, 2009.", ["meier", "2009"], "uav_communications.md"),
    ("Which ISM bands are commonly used for UAV control and video links?", "2.4 GHz and 5.8 GHz.", ["2.4", "5.8"], "uav_communications.md"),
    ("What is a FANET?", "A Flying Ad Hoc Network, a MANET subtype for UAVs.", ["flying ad hoc"], "uav_communications.md"),
    ("Is OLSR a proactive or reactive routing protocol?", "OLSR is proactive.", ["olsr", "proactive"], "uav_communications.md"),
    ("Which technique makes narrowband jamming less effective by changing carrier frequency?", "Frequency hopping spread spectrum (FHSS).", ["frequency hopping"], "uav_communications.md"),
    ("Which frequency do long-range telemetry radios typically use?", "900 MHz or 433 MHz.", ["900 mhz"], "uav_communications.md"),
    # gps_denied_navigation.md
    ("Which sensors does visual-inertial odometry fuse?", "Camera images and IMU measurements.", ["camera", "imu"], "gps_denied_navigation.md"),
    ("Which state estimator does the PX4 autopilot use?", "EKF2.", ["ekf2"], "gps_denied_navigation.md"),
    ("Why does IMU-only dead reckoning drift so quickly?", "Accelerometer errors are integrated twice, so position error grows quadratically.", ["integrated twice", "quadratic"], "gps_denied_navigation.md"),
    ("What does SLAM stand for?", "Simultaneous Localization and Mapping.", ["simultaneous localization and mapping"], "gps_denied_navigation.md"),
    ("What does TERCOM match measured terrain profiles against?", "Stored terrain elevation maps.", ["elevation"], "gps_denied_navigation.md"),
    ("Which sensors give velocity estimates close to the ground?", "Optical flow sensors.", ["optical flow"], "gps_denied_navigation.md"),
    # edge_ai_uav.md
    ("How much smaller is an INT8 model compared with FP32?", "Roughly 4x smaller.", ["4x"], "edge_ai_uav.md"),
    ("What compression ratio does 4-bit quantization give relative to FP32?", "Roughly 8x.", ["8x"], "edge_ai_uav.md"),
    ("What are the two main quantization approaches?", "Post-training quantization and quantization-aware training.", ["post-training", "quantization-aware"], "edge_ai_uav.md"),
    ("What is knowledge distillation?", "Training a small student model to mimic a larger teacher model.", ["student", "teacher"], "edge_ai_uav.md"),
    ("Which inference optimizer targets NVIDIA GPUs?", "TensorRT.", ["tensorrt"], "edge_ai_uav.md"),
    ("Why does onboard compute reduce UAV flight time?", "It competes with propulsion for battery capacity.", ["battery"], "edge_ai_uav.md"),
    # mission_data_analytics.md
    ("Which two flight-log formats are common?", "PX4 ULog and ArduPilot DataFlash.", ["ulog", "dataflash"], "mission_data_analytics.md"),
    ("Which tool is used to ingest streaming telemetry in a typical pipeline?", "Apache Kafka.", ["kafka"], "mission_data_analytics.md"),
    ("What does increased motor current draw often indicate?", "Motor degradation.", ["motor degradation"], "mission_data_analytics.md"),
    ("Name anomaly detection techniques for telemetry.", "Z-score on residuals, isolation forest, autoencoders.", ["isolation forest", "autoencoder"], "mission_data_analytics.md"),
    ("Which table formats store curated telemetry data?", "Parquet or Delta Lake.", ["parquet"], "mission_data_analytics.md"),
    ("Which KPIs do teams track for mission performance?", "Success rate, area coverage per hour, battery per km, time-to-complete.", ["success rate", "coverage"], "mission_data_analytics.md"),
]

# Paraphrased (lexically distant) versions of every second document question: a harder retrieval test.
# key = 1-based index into DOC_QS. Ground truth and keywords stay the same.
PARAPHRASE = {
    2: "Which two scientists published the velocity-update optimizer inspired by bird flocking, and in which year?",
    4: "How does the runtime of the classical optimal assignment solver scale with the number of tasks?",
    6: "Whose pheromone-trail inspired route-finding technique appeared in the early 1990s?",
    8: "Which unlicensed radio frequencies do hobby and commercial drones typically use for piloting and live video?",
    10: "Which UAV routing protocol keeps routes updated all the time rather than on demand?",
    12: "What lower radio frequencies are chosen for long-distance drone telemetry links?",
    14: "Which sensor-fusion filter implementation runs inside the PX4 flight stack?",
    16: "What technique lets a robot map its surroundings while localizing itself within that map?",
    18: "Which downward-looking sensors estimate ground speed at low altitude?",
    20: "How much do weights shrink when stored with four bits rather than full precision?",
    22: "What technique transfers knowledge from a big network to a smaller one?",
    24: "Why do heavier onboard computers shorten mission endurance?",
    26: "Which distributed event-streaming platform ingests telemetry in a typical pipeline?",
    28: "How can unusual telemetry samples be flagged automatically?",
    30: "What indicators show how well missions are performing?",
}

# (question, sql returning one value)
SQL_QS = [
    ("In the missions database, how many missions are recorded?", "SELECT COUNT(*) FROM missions"),
    ("In the missions database, what is the average mission duration in minutes?", "SELECT ROUND(AVG(duration_min),1) FROM missions"),
    ("In the missions database, what is the average battery used percent per mission?", "SELECT ROUND(AVG(battery_used_pct),1) FROM missions"),
    ("In the missions database, what is the overall mission success rate in percent?", "SELECT ROUND(100.0*SUM(success)/COUNT(*),1) FROM missions"),
    ("In the missions database, what is the total area covered in km2?", "SELECT ROUND(SUM(area_km2),1) FROM missions"),
    ("In the missions database, what is the maximum number of UAVs in a single mission?", "SELECT MAX(uav_count) FROM missions"),
    ("In the missions database, how many missions were flown over desert terrain?", "SELECT COUNT(*) FROM missions WHERE terrain='desert'"),
    ("In the missions database, how many missions were flown over mountain terrain?", "SELECT COUNT(*) FROM missions WHERE terrain='mountain'"),
    ("In the missions database, what is the average battery used percent for mountain missions?", "SELECT ROUND(AVG(battery_used_pct),1) FROM missions WHERE terrain='mountain'"),
    ("In the missions database, what is the average battery used percent for urban missions?", "SELECT ROUND(AVG(battery_used_pct),1) FROM missions WHERE terrain='urban'"),
    ("In the missions database, what is the success rate in percent of missions using the cbba algorithm?", "SELECT ROUND(100.0*SUM(success)/COUNT(*),1) FROM missions WHERE swarm_algorithm='cbba'"),
    ("In the missions database, what is the success rate in percent of missions using the leader_follower algorithm?", "SELECT ROUND(100.0*SUM(success)/COUNT(*),1) FROM missions WHERE swarm_algorithm='leader_follower'"),
    ("In the missions database, how many missions used the pso algorithm?", "SELECT COUNT(*) FROM missions WHERE swarm_algorithm='pso'"),
    ("In the missions database, how many anomalies are recorded in total?", "SELECT COUNT(*) FROM anomalies"),
    ("In the missions database, which anomaly type is the most common?", "SELECT type FROM anomalies GROUP BY type ORDER BY COUNT(*) DESC, type LIMIT 1"),
    ("In the missions database, how many anomalies have severity 3?", "SELECT COUNT(*) FROM anomalies WHERE severity=3"),
    ("In the missions database, how many distinct missions had at least one anomaly?", "SELECT COUNT(DISTINCT mission_id) FROM anomalies"),
    ("In the missions database, what is the average UAV count for forest missions?", "SELECT ROUND(AVG(uav_count),1) FROM missions WHERE terrain='forest'"),
    ("In the missions database, how many missions failed?", "SELECT COUNT(*) FROM missions WHERE success=0"),
    ("In the missions database, how many missions lasted more than 60 minutes?", "SELECT COUNT(*) FROM missions WHERE duration_min>60"),
]


def fmt(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def main() -> None:
    db = sqlite3.connect(ROOT / "data" / "sample.db")
    rows = []
    for i, (q, gt, kw, src) in enumerate(DOC_QS, 1):
        style = "paraphrased" if i in PARAPHRASE else "direct"
        rows.append({
            "id": f"doc-{i:02d}", "type": "doc", "style": style, "question": PARAPHRASE.get(i, q),
            "ground_truth": gt, "keywords": kw, "expected_source": src,
        })
    for i, (q, sql) in enumerate(SQL_QS, 1):
        val = fmt(db.execute(sql).fetchone()[0])
        rows.append({"id": f"sql-{i:02d}", "type": "sql", "question": q, "ground_truth": val, "keywords": [val.lower()], "expected_source": None, "sql": sql})
    out = ROOT / "data" / "eval" / "golden_set.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} questions -> {out}")


if __name__ == "__main__":
    main()
