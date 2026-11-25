# G1 épée simulation model

The model uses a 29-DoF Unitree G1 with a fixed right-hand BrainCo Revo2 grasp and épée. The grasp is obtained by aligning the weapon axis in the human wrist frame to the robot wrist frame and optimizing hand–handle contact. The fingers remain fixed during motion imitation.

- `robot/grasp/`: grasp specification, inertial properties and meshes for wrist integration.
- `robot/g1_revo2_epee_menagerie/`: standalone MJCF for geometric checks and replay.
- `robot/evidence/`: numerical grasp and model validation records.
- `fencingrobot/robot_spec.py`: integration with mjlab G1, including weapon-tip sites and baseline soft-contact approximation.

In the training model, the hand and weapon are merged into `right_wrist_yaw_link` without adding controlled joints. The root-plus-joint configuration therefore has 36 entries. Actuator limits follow the stock mjlab G1 configuration, including the 5 N·m wrist pitch and yaw limits. The baseline approximates blade compliance through segment-specific contact parameters rather than simulating a deformable blade.

The mounting geometry, handle dimensions and weapon inertia are modeling assumptions, not measurements of a physical assembled robot. The model is intended for simulation research; no hardware safety or deployment validation is claimed.

Third-party notices are retained in `robot/LICENSES/`: Unitree Robotics G1 assets and BrainCo Revo2 assets (Apache-2.0). See the included notices for their actual terms.
