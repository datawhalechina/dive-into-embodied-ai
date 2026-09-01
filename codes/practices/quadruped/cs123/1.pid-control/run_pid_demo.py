"""带实时曲线和参数调节的 PID 控制演示。

运行：
    python run_pid_demo.py

依赖：
    pip install matplotlib
"""

from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider


class PID:
    """带输出限幅和简单积分抗饱和的离散 PID 控制器。"""

    def __init__(self, kp, ki, kd, output_limits=(0.0, 2.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output, self.max_output = output_limits
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.last_error = None

    def update(self, setpoint, measurement, dt):
        if dt <= 0:
            raise ValueError("dt 必须大于 0")

        error = setpoint - measurement
        derivative = (
            0.0
            if self.last_error is None
            else (error - self.last_error) / dt
        )

        # 先尝试更新积分，再判断是否发生输出饱和。
        candidate_integral = self.integral + error * dt
        raw_output = (
            self.kp * error
            + self.ki * candidate_integral
            + self.kd * derivative
        )
        output = max(self.min_output, min(self.max_output, raw_output))

        # 未饱和，或误差有助于退出饱和区时，才继续累积积分。
        leaving_upper_limit = raw_output > self.max_output and error < 0
        leaving_lower_limit = raw_output < self.min_output and error > 0
        if output == raw_output or leaving_upper_limit or leaving_lower_limit:
            self.integral = candidate_integral

        self.last_error = error
        return output


class SecondOrderPlant:
    """用于演示 PID 效果的二阶被控对象。"""

    def __init__(self, natural_frequency=2.2, damping_ratio=0.48):
        self.natural_frequency = natural_frequency
        self.damping_ratio = damping_ratio
        self.reset()

    def reset(self):
        self.value = 0.0
        self.velocity = 0.0

    def update(self, control, dt):
        wn = self.natural_frequency
        zeta = self.damping_ratio

        acceleration = (
            wn**2 * (control - self.value)
            - 2.0 * zeta * wn * self.velocity
        )
        self.velocity += acceleration * dt
        self.value += self.velocity * dt
        return self.value


# -------------------- 仿真参数 --------------------

DT = 0.02
DISPLAY_SECONDS = 15.0
MAX_POINTS = int(DISPLAY_SECONDS / DT) + 1

pid = PID(kp=1.8, ki=0.8, kd=0.25)
plant = SecondOrderPlant()

simulation_time = 0.0
control_output = 0.0
running = True

times = deque([0.0], maxlen=MAX_POINTS)
setpoints = deque([1.0], maxlen=MAX_POINTS)
measurements = deque([0.0], maxlen=MAX_POINTS)
outputs = deque([0.0], maxlen=MAX_POINTS)


# -------------------- 图形布局 --------------------

# 尽量使用系统中的中文字体。
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Arial Unicode MS",
    "Heiti SC",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

fig, (ax_value, ax_output) = plt.subplots(
    2,
    1,
    figsize=(10, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

window_manager = getattr(fig.canvas, "manager", None)
if window_manager and hasattr(window_manager, "set_window_title"):
    window_manager.set_window_title("PID 实时控制演示")

fig.subplots_adjust(
    left=0.10,
    right=0.96,
    top=0.93,
    bottom=0.30,
    hspace=0.14,
)

target_line, = ax_value.plot(
    [],
    [],
    "--",
    linewidth=2,
    label="目标值",
)
value_line, = ax_value.plot(
    [],
    [],
    linewidth=2.5,
    label="过程值",
)
output_line, = ax_output.plot(
    [],
    [],
    linewidth=2,
    color="tab:green",
    label="控制输出",
)

ax_value.set_title("PID 控制实时响应")
ax_value.set_ylabel("目标 / 过程值")
ax_value.set_ylim(0.0, 1.8)
ax_value.grid(True, alpha=0.25)
ax_value.legend(loc="upper left")

ax_output.set_xlabel("时间（秒）")
ax_output.set_ylabel("输出")
ax_output.set_ylim(-0.05, 2.05)
ax_output.grid(True, alpha=0.25)
ax_output.legend(loc="upper left")

status_text = ax_value.text(
    0.99,
    0.97,
    "",
    transform=ax_value.transAxes,
    horizontalalignment="right",
    verticalalignment="top",
)


# -------------------- 交互控件 --------------------

kp_slider = Slider(
    fig.add_axes([0.12, 0.215, 0.32, 0.03]),
    "Kp",
    0.0,
    6.0,
    valinit=1.8,
    valstep=0.05,
)
ki_slider = Slider(
    fig.add_axes([0.58, 0.215, 0.32, 0.03]),
    "Ki",
    0.0,
    4.0,
    valinit=0.8,
    valstep=0.05,
)
kd_slider = Slider(
    fig.add_axes([0.12, 0.165, 0.32, 0.03]),
    "Kd",
    0.0,
    2.0,
    valinit=0.25,
    valstep=0.01,
)
setpoint_slider = Slider(
    fig.add_axes([0.58, 0.165, 0.32, 0.03]),
    "目标值",
    0.4,
    1.6,
    valinit=1.0,
    valstep=0.05,
)

pause_button = Button(
    fig.add_axes([0.34, 0.065, 0.12, 0.055]),
    "暂停",
)
reset_button = Button(
    fig.add_axes([0.54, 0.065, 0.12, 0.055]),
    "复位",
)


def reset_simulation(_event=None):
    """复位被控对象和 PID 内部状态，保留当前滑块参数。"""
    global simulation_time, control_output

    simulation_time = 0.0
    control_output = 0.0
    pid.reset()
    plant.reset()

    times.clear()
    setpoints.clear()
    measurements.clear()
    outputs.clear()

    setpoint = setpoint_slider.val
    times.append(0.0)
    setpoints.append(setpoint)
    measurements.append(0.0)
    outputs.append(0.0)


def toggle_pause(_event=None):
    """暂停或继续仿真。"""
    global running

    running = not running
    pause_button.label.set_text("暂停" if running else "继续")


def animate(_frame):
    """推进一次仿真，并刷新曲线。"""
    global simulation_time, control_output

    if running:
        pid.kp = kp_slider.val
        pid.ki = ki_slider.val
        pid.kd = kd_slider.val
        setpoint = setpoint_slider.val

        control_output = pid.update(setpoint, plant.value, DT)
        measurement = plant.update(control_output, DT)
        simulation_time += DT

        times.append(simulation_time)
        setpoints.append(setpoint)
        measurements.append(measurement)
        outputs.append(control_output)

    time_data = list(times)
    target_data = list(setpoints)
    measurement_data = list(measurements)
    output_data = list(outputs)

    target_line.set_data(time_data, target_data)
    value_line.set_data(time_data, measurement_data)
    output_line.set_data(time_data, output_data)

    right_edge = max(DISPLAY_SECONDS, simulation_time)
    ax_value.set_xlim(
        right_edge - DISPLAY_SECONDS,
        right_edge,
    )

    error = setpoint_slider.val - plant.value
    status_text.set_text(
        f"过程值 {plant.value:.3f}\n"
        f"误差 {error:.3f}\n"
        f"输出 {control_output:.3f}"
    )

    return target_line, value_line, output_line, status_text


pause_button.on_clicked(toggle_pause)
reset_button.on_clicked(reset_simulation)

# 必须保存动画对象，否则可能会被垃圾回收而停止刷新。
animation = FuncAnimation(
    fig,
    animate,
    interval=int(DT * 1000),
    blit=False,
    cache_frame_data=False,
)

plt.show()
