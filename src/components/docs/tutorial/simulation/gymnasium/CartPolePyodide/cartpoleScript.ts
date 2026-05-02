// 在 Pyodide 中真实执行的 Python 脚本。
// 预先注入 env 对象 + env_init / env_step 两个函数,
// JS 侧调用时只调用函数,不每次重新编译字符串,减少跨桥开销。
export const CARTPOLE_RUNTIME_SCRIPT = `
import gymnasium as gym

_env = None

def env_init(seed):
    global _env
    if _env is not None:
        try:
            _env.close()
        except Exception:
            pass
    _env = gym.make("CartPole-v1")
    obs, _info = _env.reset(seed=int(seed))
    return obs.tolist()

def env_step():
    global _env
    action = int(_env.action_space.sample())
    obs, reward, terminated, truncated, _info = _env.step(action)
    return {
        "obs": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "action": action,
    }

def env_close():
    global _env
    if _env is not None:
        try:
            _env.close()
        except Exception:
            pass
        _env = None
`;

// 展示给读者看的代码(只读),和章节原版示例相比去掉了 render_mode。
export const CARTPOLE_DISPLAY_SCRIPT = `import gymnasium as gym

# 浏览器里运行时去掉了 render_mode="human"
# 因为 pygame 无法在 Pyodide 中启用,改由页面上的 Canvas 负责渲染
env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)
total_reward = 0.0
episode_over = False

while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
`;
