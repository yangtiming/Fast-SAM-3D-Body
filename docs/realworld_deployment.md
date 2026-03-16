# Real-World Deployment

**Note:** Pose estimation may occasionally fail and produce jerky motions. Always validate in simulation and confirm with the debug scripts before deploying to real hardware.

## Prerequisites

Download the following assets and place them somewhere accessible:

- **SMPL model** from [SMPLify](https://smplify.is.tue.mpg.de/)
- **MHR conversion assets** from [MHR tools](https://github.com/facebookresearch/MHR/tree/main/tools/mhr_smpl_conversion/assets)

This deployment uses [SONIC](https://nvlabs.github.io/GR00T-WholeBodyControl/) for robot control. Please follow the [SONIC installation guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html) to set up the environment.

## Running the Publisher

Start the pose publisher, which reads from a camera (or recorded video), runs Fast SAM 3D Body inference, and broadcasts SMPL pose over ZMQ.

### Single-View (single camera)

```bash
python run_publisher.py \
    --source camera \
    --smpl-model-path path/to/SMPL_NEUTRAL.pkl \
    --nn-model-dir mhr2smpl/experiments/multiview_n30000_e500 \
    --mhr2smpl-mapping-path path/to/mhr2smpl_mapping.npz \
    --mhr-mesh-path path/to/mhr_face_mask.ply \
    --smoother-dir mhr2smpl/experiments/smoother_w5 \
    --record
```

### Multi-View (multiple RealSense cameras)

```bash
python run_multiview_publisher.py \
    --source camera \
    --serials <serial_0>,<serial_1> \
    --smpl-model-path path/to/SMPL_NEUTRAL.pkl \
    --nn-model-dir mhr2smpl/experiments/multiview_n30000_e500 \
    --mhr2smpl-mapping-path path/to/mhr2smpl_mapping.npz \
    --mhr-mesh-path path/to/mhr_face_mask.ply \
    --smoother-dir mhr2smpl/experiments/smoother_w5 \
    --record
```

`--serials` is required. Pass RealSense serial numbers comma-separated; the first serial is treated as the main camera.

## Running SONIC

The publisher streams SMPL-based poses over ZMQ using [Protocol v2](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/zmq.html#protocol-v2-smpl-based-encode-mode-2). Start SONIC pointing to the observation config below.

```bash
# Simulation
bash deploy.sh sim --input-type zmq --obs-config path/to/obs_config.yaml

# Real robot
bash deploy.sh real --input-type zmq --obs-config path/to/obs_config.yaml
```

<details>
<summary><strong>Observation config</strong></summary>

```yaml
observations:

  - name: "token_state"
    enabled: true

  - name: "his_base_angular_velocity_10frame_step1"
    enabled: true

  - name: "his_body_joint_positions_10frame_step1"
    enabled: true

  - name: "his_body_joint_velocities_10frame_step1"
    enabled: true

  - name: "his_last_actions_10frame_step1"
    enabled: true

  - name: "his_gravity_dir_10frame_step1"
    enabled: true

encoder:
  dimension: 64
  use_fp16: false
  encoder_observations:
    - name: "encoder_mode_4"
      enabled: true
    - name: "motion_joint_positions_10frame_step5"
      enabled: true
    - name: "motion_joint_velocities_10frame_step5"
      enabled: true
    - name: "motion_root_z_position_10frame_step5"
      enabled: true
    - name: "motion_root_z_position"
      enabled: true
    - name: "motion_anchor_orientation"
      enabled: true
    - name: "motion_anchor_orientation_10frame_step5"
      enabled: true
    - name: "motion_joint_positions_lowerbody_10frame_step5"
      enabled: true
    - name: "motion_joint_velocities_lowerbody_10frame_step5"
      enabled: true
    - name: "vr_3point_local_target"
      enabled: true
    - name: "vr_3point_local_orn_target"
      enabled: true
    - name: "smpl_joints_10frame_step1"
      enabled: true
    - name: "smpl_anchor_orientation_10frame_step1"
      enabled: true
    - name: "motion_joint_positions_wrists_10frame_step1"
      enabled: true
  encoder_modes:
    - name: "g1"
      mode_id: 0
      required_observations:
        - encoder_mode_4
        - motion_joint_positions_10frame_step5
        - motion_joint_velocities_10frame_step5
        - motion_anchor_orientation_10frame_step5
    - name: "teleop"
      mode_id: 1
      required_observations:
        - encoder_mode_4
        - motion_joint_positions_lowerbody_10frame_step5
        - motion_joint_velocities_lowerbody_10frame_step5
        - vr_3point_local_target
        - vr_3point_local_orn_target
        - motion_anchor_orientation
    - name: "smpl"
      mode_id: 2
      required_observations:
        - encoder_mode_4
        - smpl_joints_10frame_step1
        - smpl_anchor_orientation_10frame_step1
```

</details>

## Recording and Debugging

To capture video for offline replay, use `record_realsense.py` or `record_realsense_multi.py`.

To verify the pose stream visually, `debug_smpl_stream.py` subscribes to the publisher and renders a front/side-view SMPL mesh video.

```bash
python debug_smpl_stream.py \
    --smpl-model-path path/to/SMPL_NEUTRAL.pkl \
    --num-frames 300 \
    --render-output output/smpl_debug.mp4 \
    --show-joints
```
