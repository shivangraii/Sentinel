# UAV Swarm Communications

Reliable communication links are the backbone of any swarm. Vehicles exchange telemetry, commands and sometimes video, often while moving fast and while the radio environment is contested.

## Protocols and bands

MAVLink is a lightweight messaging protocol for communicating with drones and between onboard components. It was created by Lorenz Meier and first released in 2009. Control and video links commonly use the 2.4 GHz and 5.8 GHz ISM bands, while long-range telemetry radios typically operate at 900 MHz or 433 MHz because lower frequencies travel farther and penetrate obstacles better.

## Ad hoc networking

A mobile ad hoc network (MANET) lets nodes relay traffic for each other without fixed infrastructure. A Flying Ad Hoc Network (FANET) is a MANET subtype designed for UAVs, where node mobility is very high and topology changes quickly. OLSR is a proactive routing protocol that keeps routes up to date continuously, whereas AODV is a reactive protocol that discovers routes only when traffic needs them. Geographic routing such as GPSR forwards packets using node positions.

## Resilience

Jamming is a major threat. Frequency hopping spread spectrum (FHSS) rapidly changes the carrier frequency according to a shared pattern, which makes narrowband jamming much less effective. Direct-sequence spread spectrum spreads the signal over a wider band to improve resistance to interference. Closed-loop control needs low latency, so designers trade bandwidth against delay when choosing what to send over the swarm network.
