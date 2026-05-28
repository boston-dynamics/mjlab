import colorsys
import dataclasses
from typing import Any, Literal

import mujoco

from mjlab.asset_zoo.robots import (
  YAM_ACTION_SCALE,
  get_yam_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  BuiltinSensorCfg,
  CameraSensorCfg,
  ContactSensorCfg,
  ObjRef,
)
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import (
  LiftingCommandCfg,
  MultiCubeLiftingCommandCfg,
  SensorLiftingCommandCfg,
  SensorMultiCubeLiftingCommandCfg,
  sensor_bring_object_reward,
  sensor_cube_to_goal_distance,
  sensor_ee_to_cube_distance,
  sensor_multi_cube_staged_position_reward,
  sensor_staged_position_reward,
  sensor_target_position,
)

# Sensor names for sensor-based task variants

_CUBE_POS = "cube_pos"
_ROOT_QUAT = "root_quat"
_GRASP_POS = "grasp_pos"
_GRASP_QUAT = "grasp_quat"

_CUBE_POS_SENSOR = f"cube/{_CUBE_POS}"
_ROOT_QUAT_SENSOR = f"robot/{_ROOT_QUAT}"
_GRASP_POS_SENSOR = f"robot/{_GRASP_POS}"
_GRASP_QUAT_SENSOR = f"robot/{_GRASP_QUAT}"


def get_cube_spec(
  cube_size: float = 0.02,
  mass: float = 0.05,
  rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=rgba,
  )
  return spec


def yam_lift_cube_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_yam_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "grasp_site",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("grasp_site",)

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensor pattern.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"

  cfg.viewer.body_name = "arm"

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

    # Higher command resampling frequency for more dynamic play.
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

  return cfg


def yam_lift_cube_mujoco_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """YAM lift-cube task variant using sensors, compatible with MJWarp and MuJoCo.

  Returns:
      Configured ManagerBasedRlEnvCfg.
  """
  cfg = yam_lift_cube_env_cfg(play=play)

  # --- Sensors (backend-agnostic kinematics via sensordata) ---
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    BuiltinSensorCfg(
      name=_GRASP_POS,
      sensor_type="framepos",
      obj=ObjRef(type="site", name="grasp_site", entity="robot"),
    ),
    BuiltinSensorCfg(
      name=_CUBE_POS,
      sensor_type="framepos",
      obj=ObjRef(type="body", name="cube", entity="cube"),
    ),
    BuiltinSensorCfg(
      name=_ROOT_QUAT,
      sensor_type="framequat",
      obj=ObjRef(type="body", name="arm", entity="robot"),
    ),
  )

  # --- Observations ---
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    existing_ee = terms["ee_to_cube"]
    existing_cube_to_goal = terms["cube_to_goal"]
    terms["ee_to_cube"] = ObservationTermCfg(
      func=sensor_ee_to_cube_distance,
      params={
        "grasp_sensor_name": _GRASP_POS_SENSOR,
        "cube_sensor_name": _CUBE_POS_SENSOR,
        "base_quat_sensor_name": _ROOT_QUAT_SENSOR,
      },
      noise=existing_ee.noise,
    )
    terms["cube_to_goal"] = ObservationTermCfg(
      func=sensor_cube_to_goal_distance,
      params={
        "command_name": existing_cube_to_goal.params["command_name"],
        "cube_sensor_name": _CUBE_POS_SENSOR,
        "base_quat_sensor_name": _ROOT_QUAT_SENSOR,
      },
      noise=existing_cube_to_goal.noise,
    )

  # --- Rewards ---
  existing_lift = cfg.rewards["lift"].params
  cfg.rewards["lift"].func = sensor_staged_position_reward
  cfg.rewards["lift"].params = {
    "command_name": existing_lift["command_name"],
    "grasp_sensor_name": _GRASP_POS_SENSOR,
    "cube_sensor_name": _CUBE_POS_SENSOR,
    "reaching_std": existing_lift["reaching_std"],
    "bringing_std": existing_lift["bringing_std"],
  }
  existing_lift_precise = cfg.rewards["lift_precise"].params
  cfg.rewards["lift_precise"].func = sensor_bring_object_reward
  cfg.rewards["lift_precise"].params = {
    "command_name": existing_lift_precise["command_name"],
    "cube_sensor_name": _CUBE_POS_SENSOR,
    "std": existing_lift_precise["std"],
  }

  # --- Override LiftingCommand to use sensor-based _update_metrics ---
  assert cfg.commands is not None
  existing_cmd = cfg.commands["lift_height"]
  assert isinstance(existing_cmd, LiftingCommandCfg)
  # Use dataclasses.fields to copy all fields so new fields added to
  # LiftingCommandCfg are automatically propagated without updates here.
  cfg.commands["lift_height"] = SensorLiftingCommandCfg(
    **{f.name: getattr(existing_cmd, f.name) for f in dataclasses.fields(existing_cmd)},
    cube_sensor_name=_CUBE_POS_SENSOR,
  )

  return cfg


def yam_lift_cube_vision_env_cfg(
  cam_type: Literal["rgb", "depth"],
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = yam_lift_cube_env_cfg(play=play)

  camera_names = ["robot/camera_d405"]
  cam_kwargs = {
    "robot/camera_d405": {
      "height": 32,
      "width": 32,
    },
  }
  shared_cam_kwargs = dict(
    data_types=(cam_type,),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )

  cam_terms = {}
  for cam_name in camera_names:
    cam_cfg = CameraSensorCfg(
      name=cam_name.split("/")[-1],
      camera_name=cam_name,
      **cam_kwargs[cam_name],  # type: ignore[invalid-argument-type]
      **shared_cam_kwargs,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)
    param_kwargs: dict[str, Any] = {"sensor_name": cam_cfg.name}
    if cam_type == "depth":
      param_kwargs["cutoff_distance"] = 0.5
      func = manipulation_mdp.camera_depth
    else:
      func = manipulation_mdp.camera_rgb
    cam_terms[f"{cam_name.split('/')[-1]}_{cam_type}"] = ObservationTermCfg(
      func=func, params=param_kwargs
    )

  camera_obs = ObservationGroupCfg(
    terms=cam_terms, enable_corruption=False, concatenate_terms=True
  )
  cfg.observations["camera"] = camera_obs

  if cam_type == "rgb":
    cfg.events["cube_color"] = EventTermCfg(
      func=dr.geom_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("cube", geom_names=(".*",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": (0.0, 1.0),
      },
    )

  # Pop privileged info from actor observations.
  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("ee_to_cube")
  actor_obs.terms.pop("cube_to_goal")

  # Add goal_position to actor observations.
  actor_obs.terms["goal_position"] = ObservationTermCfg(
    func=manipulation_mdp.target_position,
    params={
      "command_name": "lift_height",
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
    # NOTE: No noise for goal position.
  )

  return cfg


def yam_lift_cube_mujoco_vision_env_cfg(
  cam_type: Literal["rgb", "depth"] = "rgb",
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """YAM lift-cube vision task variant using sensors, compatible with MJWarp and MuJoCo.

  Combines the sensor-based observations/rewards from yam_lift_cube_mujoco_env_cfg
  with the camera-based actor observation pipeline from yam_lift_cube_vision_env_cfg.

  Returns:
      Configured ManagerBasedRlEnvCfg.
  """
  cfg = yam_lift_cube_mujoco_env_cfg(play=play)

  camera_names = ["robot/camera_d405"]
  cam_kwargs = {
    "robot/camera_d405": {
      "height": 32,
      "width": 32,
    },
  }
  shared_cam_kwargs = dict(
    data_types=(cam_type,),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )

  cam_terms = {}
  for cam_name in camera_names:
    cam_cfg = CameraSensorCfg(
      name=cam_name.split("/")[-1],
      camera_name=cam_name,
      **cam_kwargs[cam_name],  # type: ignore[invalid-argument-type]
      **shared_cam_kwargs,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)
    param_kwargs: dict[str, Any] = {"sensor_name": cam_cfg.name}
    if cam_type == "depth":
      param_kwargs["cutoff_distance"] = 0.5
      func = manipulation_mdp.camera_depth
    else:
      func = manipulation_mdp.camera_rgb
    cam_terms[f"{cam_name.split('/')[-1]}_{cam_type}"] = ObservationTermCfg(
      func=func, params=param_kwargs
    )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    BuiltinSensorCfg(
      name=_GRASP_QUAT,
      sensor_type="framequat",
      obj=ObjRef(type="site", name="grasp_site", entity="robot"),
    ),
  )

  cfg.observations["camera"] = ObservationGroupCfg(
    terms=cam_terms, enable_corruption=False, concatenate_terms=True
  )

  # Actor uses vision + goal in place of privileged ee_to_cube / cube_to_goal.
  # Critic retains the sensor-based privileged terms inherited from the mujoco cfg.
  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("ee_to_cube")
  actor_obs.terms.pop("cube_to_goal")
  actor_obs.terms["goal_position"] = ObservationTermCfg(
    func=sensor_target_position,
    params={
      "command_name": "lift_height",
      "grasp_sensor_name": _GRASP_POS_SENSOR,
      "base_quat_sensor_name": _GRASP_QUAT_SENSOR,
    },
    # NOTE: No noise for goal position.
  )

  return cfg


def _cube_color(i: int, n: int) -> tuple[float, float, float, float]:
  """Generate a distinct color for cube i of n using HSV hue rotation."""
  h = i / max(n, 1)
  r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
  return (r, g, b, 1.0)


def yam_multi_cube_seg_env_cfg(
  num_cubes: int = 3,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Multi-cube task: depth + segmentation mask for goal conditioning."""
  cfg = make_lift_cube_env_cfg()

  cube_names = [f"cube_{i}" for i in range(num_cubes)]
  entities: dict[str, EntityCfg] = {"robot": get_yam_robot_cfg()}
  for i, name in enumerate(cube_names):
    color = _cube_color(i, num_cubes)
    entities[name] = EntityCfg(
      spec_fn=lambda c=color: get_cube_spec(rgba=c),
    )
  cfg.scene.entities = entities

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  cfg.commands = {
    "lift_height": MultiCubeLiftingCommandCfg(
      entity_names=tuple(cube_names),
      resampling_time_range=(8.0, 12.0),
      debug_vis=True,
      difficulty="dynamic",
    ),
  }

  cfg.rewards["lift"] = RewardTermCfg(
    func=manipulation_mdp.multi_cube_staged_position_reward,
    weight=1.0,
    params={
      "command_name": "lift_height",
      "reaching_std": 0.2,
      "bringing_std": 0.3,
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
  )
  cfg.rewards["lift_precise"] = RewardTermCfg(
    func=manipulation_mdp.multi_cube_bring_object_reward,
    weight=1.0,
    params={
      "command_name": "lift_height",
      "std": 0.05,
    },
  )

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"

  cfg.viewer.body_name = "arm"
  cfg.sim.nconmax = max(cfg.sim.nconmax or 55, 55 + num_cubes * 120)

  cam_cfg = CameraSensorCfg(
    name="camera_d405",
    camera_name="robot/camera_d405",
    height=32,
    width=32,
    data_types=("depth", "segmentation"),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)

  cam_terms = {
    "depth": ObservationTermCfg(
      func=manipulation_mdp.camera_depth,
      params={
        "sensor_name": "camera_d405",
        "cutoff_distance": 0.5,
      },
    ),
    "target_mask": ObservationTermCfg(
      func=manipulation_mdp.camera_target_cube_mask,
      params={
        "sensor_name": "camera_d405",
        "command_name": "lift_height",
      },
    ),
  }
  cfg.observations["camera"] = ObservationGroupCfg(
    terms=cam_terms,
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )

  for group_name in ("actor", "critic"):
    obs = cfg.observations[group_name]
    obs.terms.pop("ee_to_cube", None)
    obs.terms.pop("cube_to_goal", None)
    obs.terms["goal_position"] = ObservationTermCfg(
      func=manipulation_mdp.target_position,
      params={
        "command_name": "lift_height",
        "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
      },
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (
      4.0,
      4.0,
    )

  return cfg


def yam_multi_cube_mujoco_seg_env_cfg(
  num_cubes: int = 3,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """YAM multi-cube seg task variant using sensors, compatible with MJWarp and MuJoCo.

  Returns:
      Configured ManagerBasedRlEnvCfg.
  """
  cfg = yam_multi_cube_seg_env_cfg(num_cubes=num_cubes, play=play)

  # --- Sensors ---
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    BuiltinSensorCfg(
      name=_GRASP_POS,
      sensor_type="framepos",
      obj=ObjRef(type="site", name="grasp_site", entity="robot"),
    ),
    BuiltinSensorCfg(
      name=_ROOT_QUAT,
      sensor_type="framequat",
      obj=ObjRef(type="body", name="arm", entity="robot"),
    ),
    BuiltinSensorCfg(
      name=_GRASP_QUAT,
      sensor_type="framequat",
      obj=ObjRef(type="site", name="grasp_site", entity="robot"),
    ),
  )
  for i in range(num_cubes):
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
      BuiltinSensorCfg(
        name=f"cube_{i}_pos",
        sensor_type="framepos",
        obj=ObjRef(type="body", name="cube", entity=f"cube_{i}"),
      ),
    )

  # --- Commands ---
  assert cfg.commands is not None
  existing_cmd = cfg.commands["lift_height"]
  assert isinstance(existing_cmd, MultiCubeLiftingCommandCfg)
  cube_sensor_names = tuple(f"cube_{i}/cube_{i}_pos" for i in range(num_cubes))
  cfg.commands["lift_height"] = SensorMultiCubeLiftingCommandCfg(
    **{f.name: getattr(existing_cmd, f.name) for f in dataclasses.fields(existing_cmd)},
    cube_sensor_names=cube_sensor_names,
  )

  # --- Rewards ---
  existing_lift = cfg.rewards["lift"].params
  cfg.rewards["lift"].func = sensor_multi_cube_staged_position_reward
  cfg.rewards["lift"].params = {
    "grasp_sensor_name": _GRASP_POS_SENSOR,
    "command_name": existing_lift["command_name"],
    "reaching_std": existing_lift["reaching_std"],
    "bringing_std": existing_lift["bringing_std"],
  }
  # lift_precise reuses multi_cube_bring_object_reward unchanged:
  # isinstance(command, MultiCubeLiftingCommand) passes via inheritance.

  # --- Observations ---
  for group_name in ("actor", "critic"):
    obs = cfg.observations[group_name]
    existing_goal_pos = obs.terms["goal_position"]
    obs.terms["goal_position"] = ObservationTermCfg(
      func=sensor_target_position,
      params={
        "command_name": existing_goal_pos.params["command_name"],
        "grasp_sensor_name": _GRASP_POS_SENSOR,
        "base_quat_sensor_name": _GRASP_QUAT_SENSOR,
      },
    )

  return cfg
