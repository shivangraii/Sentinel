# Mission Data Analytics

Every flight produces telemetry that can be turned into operational insight: position, battery voltage and current, attitude, motor outputs and link quality.

## Logs and pipelines

Common flight-log formats are PX4 ULog and ArduPilot DataFlash. A typical data engineering pipeline ingests streaming telemetry with Apache Kafka, processes it with Apache Spark, stores curated data in Parquet or Delta Lake tables, and serves dashboards from those tables.

## Key performance indicators

Teams usually track mission success rate, area coverage per hour, battery consumption per kilometre and time-to-complete. Comparing these KPIs across terrain types and coordination algorithms shows which configurations perform best.

## Anomaly detection and maintenance

Telemetry anomalies can be found with statistical methods such as a z-score on model residuals, with an isolation forest, or with autoencoders that flag samples that reconstruct poorly. For predictive maintenance, increased current draw often indicates motor degradation, and rising vibration levels point to bearing or propeller problems. Early detection lets operators replace parts before a failure ends a mission.
