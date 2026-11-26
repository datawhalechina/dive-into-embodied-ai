import {hubText} from './messages';

export type Course = {
  title: string;
  to: string;
  description?: string;
  tags?: string;
  meta?: string;
  tone?: 'free' | 'gpu' | 'real';
  children?: Course[];
};
export type PracticeGroup = {id: string; title: string; lead: string; description: string; overview: string; image: string; alt: string; courses: Course[]};
export type FoundationGroup = {id: string; title: string; lead: string; description: string; project: string; courses: Course[]};

export function practiceGroups(): PracticeGroup[] {
  const t = hubText();
  return [
    {id: 'quadruped', title: t.quadruped, lead: t.quadrupedLead, description: t.quadrupedDescription,
      overview: '/docs/practices/quadruped/placeholder', image: '/img/pupper-poster.webp', alt: t.pupperAlt,
      courses: [
        {title: t.cs123, to: '/docs/practices/quadruped/cs123/intro', description: t.cs123Description, tags: 'MuJoCo · PD · PPO', meta: t.metaIntroCpu, tone: 'free'},
        {title: t.quaddle, to: '/docs/practices/quadruped/quaddle-mujoco', description: t.quaddleDescription, tags: 'MuJoCo · CPG', meta: t.metaIntroCpu, tone: 'free'},
        {title: t.go2Mjx, to: '/docs/practices/amd/unitree-rl-mjx', description: t.go2MjxDescription, tags: 'MJX · JAX · Sim2Sim', meta: t.metaGpuRocm, tone: 'gpu'},
        {title: t.quaddleVla, to: '/docs/practices/vla/quaddle', description: t.quaddleVlaDescription, tags: 'SmolVLA · CPG', meta: t.metaGpu10, tone: 'gpu'},
        {title: t.sim2sim, to: '/docs/practices/quadruped/sim2sim/placeholder', description: t.sim2simDescription, meta: t.metaGuide, tone: 'free'},
        {title: t.sim2real, to: '/docs/practices/quadruped/sim2real-guide/placeholder', description: t.sim2realDescription, meta: t.metaRealGuide, tone: 'real'},
      ]},
    {id: 'humanoid', title: t.humanoid, lead: t.humanoidLead, description: t.humanoidDescription,
      overview: '/docs/practices/humanoid/placeholder', image: '/img/unitree-g1-poster.webp', alt: t.g1Alt,
      courses: [
        {title: t.g1, to: '/docs/practices/humanoid/unitree-g1', description: t.g1Description, tags: 'G1 · LSTM · Sim2Sim', meta: t.metaAdvGpu, tone: 'gpu'},
        {title: t.microduck, to: '/docs/practices/humanoid/microduck-rl', description: t.microduckDescription, tags: 'MuJoCo Warp · PPO', meta: t.metaGpuRocm, tone: 'gpu'},
        {title: t.flamingo, to: '/docs/practices/humanoid/flamingo-isaaclab/intro', description: t.flamingoDescription, tags: 'Isaac Lab · CaT', meta: t.metaAdvGpu, tone: 'gpu'},
        {title: t.wall, to: '/docs/practices/humanoid/microduck-rl/wall-curriculum', description: t.wallDescription, meta: t.metaAdvGpu, tone: 'gpu'},
        {title: t.balanceControl, to: '/docs/practices/humanoid/balance-control/placeholder', description: t.balanceControlDescription, meta: t.metaBrowser, tone: 'free'},
        {title: t.motionTracking, to: '/docs/practices/humanoid/motion-tracking/placeholder', description: t.motionTrackingDescription, meta: t.metaGuide, tone: 'free'},
        {title: t.taskPlanning, to: '/docs/practices/humanoid/task-planning-demo/placeholder', description: t.taskPlanningDescription, meta: t.metaGuide, tone: 'free'},
        {title: t.experiments, to: '/docs/practices/humanoid/microduck-rl/experiment-notes', description: t.experimentsDescription, meta: t.metaNotes, tone: 'free'},
      ]},
    {id: 'robot-arm', title: t.arm, lead: t.armLead, description: t.armDescription,
      overview: '/docs/practices/robot-arm/placeholder', image: '/img/act-poster.webp', alt: t.actAlt,
      courses: [
        {title: t.act, to: '/docs/practices/vla/act', description: t.actDescription, tags: 'ACT · ALOHA', meta: t.metaMidGpu, tone: 'gpu'},
        {title: t.mujoco, to: '/docs/practices/robot-arm/mujoco-arm-pick-place', description: t.mujocoDescription, meta: t.metaIntroCpu, tone: 'free'},
        {title: t.ddpg, to: '/docs/practices/robot-arm/ddpg-mujoco/reacher-v5', meta: t.metaIntroCpu, tone: 'free', children: [
          {title: 'InvertedPendulum', to: '/docs/practices/robot-arm/ddpg-mujoco/invertedpendulum-v5'},
          {title: 'Reacher', to: '/docs/practices/robot-arm/ddpg-mujoco/reacher-v5'},
          {title: 'Pusher', to: '/docs/practices/robot-arm/ddpg-mujoco/pusher-v5'},
        ]},
        {title: t.recap, to: '/docs/practices/vla/recap', description: t.recapDescription, meta: t.metaAdvGpu, tone: 'gpu'},
        {title: t.ros2Arm, to: '/docs/practices/robot-arm/ros2-arm-control/placeholder', description: t.ros2ArmDescription, meta: t.metaGuide, tone: 'free'},
        {title: t.imitationArm, to: '/docs/practices/robot-arm/imitation-learning/placeholder', description: t.imitationArmDescription, meta: t.metaMidGpu, tone: 'gpu'},
        {title: t.lerobot, to: '/docs/practices/robot-arm/data-collection/lerobot-course', description: t.lerobotDescription, meta: t.metaIntroCpu, tone: 'free'},
        {title: t.so101, to: '/docs/practices/robot-arm/data-collection/so101-lerobot-real', description: t.so101Description, meta: t.metaRealArm, tone: 'real'},
      ]},
    {id: 'amd', title: t.amd, lead: t.amdLead, description: t.amdDescription,
      overview: '/docs/practices/amd/intro', image: '/img/pupper-poster.webp', alt: t.pupperAlt,
      courses: [
        {title: t.go2Mjx, to: '/docs/practices/amd/unitree-rl-mjx', description: t.go2MjxDescription, tags: 'ROCm 10.2 · MJX · 4.5h', meta: t.metaGpuRocm, tone: 'gpu'},
        {title: t.cloud, to: '/docs/practices/amd/aup-learning-cloud', description: t.cloudDescription, meta: t.metaCloud, tone: 'free'},
        {title: t.pupperAmd, to: '/docs/practices/amd/pupper-control/intro', description: t.pupperAmdDescription, tags: 'AMD · PPO · VLA', meta: t.metaGpuRocm, tone: 'gpu'},
        {title: t.microduckAmd, to: '/docs/practices/amd/microduck-rl', description: t.microduckAmdDescription, meta: t.metaGpuRocm, tone: 'gpu'},
        {title: t.actAmd, to: '/docs/practices/amd/vla-act', description: t.actAmdDescription, meta: t.metaGpuRocm, tone: 'gpu'},
      ]},
  ];
}

export function foundationGroups(): FoundationGroup[] {
  const t = hubText();
  return [
    {id: 'decision', title: t.brain, lead: t.brainLead, description: t.brainDescription, project: '/docs/practices/vla/quaddle', courses: [
      {title: t.rl, to: '/docs/foundations/rl-for-robotics/intro'},
      {title: t.vla, to: '/docs/foundations/vla/vla-intro'},
      {title: t.world, to: '/docs/foundations/world-model/intro'},
    ]},
    {id: 'control', title: t.control, lead: t.controlLead, description: t.controlDescription, project: '/docs/practices/quadruped/cs123/intro', courses: [
      {title: t.controllers, to: '/docs/foundations/controllers/intro'},
      {title: t.ppo, to: '/docs/foundations/rl-for-robotics/ppo'},
      {title: t.planningMotion, to: '/docs/foundations/robotics-and-ros2/moveit2_basics'},
    ]},
    {id: 'perception', title: t.perception, lead: t.perceptionLead, description: t.perceptionDescription, project: '/docs/practices/robot-arm/data-collection/so101-lerobot-real', courses: [
      {title: t.vlm, to: '/docs/foundations/vlm/intro'},
      {title: t.perceptionOverview, to: '/docs/foundations/perception/placeholder'},
      {title: t.calibration, to: '/docs/foundations/perception/sensor-calibration-sim2real'},
      {title: t.stateEstimation, to: '/docs/foundations/perception/odo-timestamps-state-estimation'},
      {title: t.projection, to: '/docs/foundations/perception/sensor-projection'},
      {title: t.onlineCalibration, to: '/docs/foundations/perception/online-calibration-diagnosis'},
    ]},
    {id: 'engineering', title: t.engineering, lead: t.engineeringLead, description: t.engineeringDescription, project: '/docs/practices/robot-arm/data-collection/lerobot-course', courses: [
      {title: t.simulationTools, to: '/docs/foundations/simulation/intro'},
      {title: 'ROS2', to: '/docs/foundations/robotics-and-ros2/intro'},
      {title: t.communication, to: '/docs/foundations/communication/can-mcu'},
      {title: t.hardware, to: '/docs/foundations/hardware/placeholder'},
      {title: t.imitation, to: '/docs/foundations/rl-for-robotics/imitation-learning'},
    ]},
  ];
}
