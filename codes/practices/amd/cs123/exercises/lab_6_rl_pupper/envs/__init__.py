"""Lab 6 RL Pupper env exports.

Eager `PupperEnv` import is guarded so this package can be imported in
MJX/brax-only environments where gymnasium / stable_baselines3 are absent.
"""

try:
    from envs.pupper_env import PupperEnv  # noqa: F401
    __all__ = ["PupperEnv"]
except ImportError:
    __all__ = []
