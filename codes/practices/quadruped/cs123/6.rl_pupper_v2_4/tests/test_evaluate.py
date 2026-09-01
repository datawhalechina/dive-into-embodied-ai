from stable_baselines3 import PPO

import evaluate
from pupper_env import PupperEnv


def test_evaluate_renders_outputs(tmp_path, monkeypatch):
    env = PupperEnv()
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [32, 32]},
        device="cpu",
    )

    monkeypatch.setattr(evaluate, "CMD_SCRIPT", [((0.0, 0.0, 0.0), 0.1)])
    monkeypatch.setattr(evaluate, "GIF_FPS", 2)

    gif_path = tmp_path / "demo.gif"
    plot_path = tmp_path / "velocity_tracking.png"
    evaluate.render_demo(model, env, gif_path)
    evaluate.render_velocity(
        model,
        env,
        plot_path,
        vx_commands=(0.2,),
        seconds=0.1,
    )

    assert gif_path.exists() and gif_path.stat().st_size > 0
    assert plot_path.exists() and plot_path.stat().st_size > 0
