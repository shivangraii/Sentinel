# GPS-Denied Navigation for UAVs

GPS can be jammed, spoofed or simply unavailable indoors and in urban canyons, so UAVs need alternative ways to estimate position.

## Dead reckoning and its limits

Dead reckoning integrates inertial measurements to estimate motion. An IMU-only estimate drifts because accelerometer errors are integrated twice, so position error grows quadratically with time even for a small constant bias. This is why inertial navigation must be corrected by other sensors.

## Sensor fusion

Visual-inertial odometry (VIO) fuses camera images with IMU measurements to track motion without GPS. An Extended Kalman Filter (EKF) is the standard tool for combining sensors with different noise and update rates, and the PX4 autopilot uses the EKF2 estimator for state estimation. A barometer provides altitude, and optical flow sensors provide velocity estimates close to the ground.

## Mapping and map matching

SLAM stands for Simultaneous Localization and Mapping: the vehicle builds a map while estimating its own position inside it. LiDAR-based odometry methods such as LOAM estimate motion from laser scans. Terrain-aided navigation, for example TERCOM, matches measured terrain height profiles against stored terrain elevation maps to correct position without GPS.
