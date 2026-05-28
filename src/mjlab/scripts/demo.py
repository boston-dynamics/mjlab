"""Script to run a tracking demo with a pretrained policy.

This demo downloads a pretrained checkpoint and motion file from cloud storage
and launches an interactive viewer with a humanoid robot performing a cartwheel.
"""

import tyro

import mjlab
from mjlab.scripts.gcs import ensure_default_checkpoint, ensure_default_motion
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks.registry import load_env_cfg

_TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1"


def main() -> None:
  """Run demo with pretrained tracking policy."""
  print("🎮 Setting up mjlab demo with pretrained tracking policy...")

  try:
    checkpoint_path = ensure_default_checkpoint()
    motion_path = ensure_default_motion()
  except RuntimeError as e:
    print(f"❌ Failed to download demo assets: {e}")
    print("Please check your internet connection and try again.")
    return

  env_cfg = load_env_cfg(_TASK_ID, play=True)
  env_cfg.scene.num_envs = 8

  args = tyro.cli(
    PlayConfig,
    default=PlayConfig(
      env=env_cfg,
      checkpoint_file=checkpoint_path,
      motion_file=motion_path,
      viewer="viser",
      _demo_mode=True,
    ),
    config=mjlab.TYRO_FLAGS,
  )
  run_play(_TASK_ID, args)


if __name__ == "__main__":
  main()
