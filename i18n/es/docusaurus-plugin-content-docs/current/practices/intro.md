---
title: "Proyectos: de la simulación a los robots reales"
description: "Elige proyectos por plataforma, simulación y despliegue en robots reales."
sidebar_position: 1
displayed_sidebar: practicesOverviewSidebar
---
# Proyectos: de la simulación a los robots reales

Elige una tarea y recorre entorno, entrenamiento, control y evaluación con un proyecto, una demo o una reproducción. Sigue capítulos para construir un sistema o un experimento independiente para probar un método. Consulta los [fundamentos](/docs/foundations/intro) para los principios y los [recursos](/docs/introduction/resources) para artículos, datos y código.

## Por dónde empezar \{#getting-started}

- Desde cero: el [proyecto de cuadrúpedos](/docs/practices/quadruped/cs123/intro) tiene ocho capítulos de actuadores, cinemática, modelado, marchas y políticas. Necesitas Python básico; puedes aprender control y álgebra lineal durante la práctica.
- Datos y hardware de brazos: [apuntes LeRobot en chino](/docs/practices/robot-arm/data-collection/lerobot-course), unidades 0–2 y Python básico, después [SO-101](/docs/practices/robot-arm/data-collection/so101-lerobot-real), que requiere un robot compatible.
- Primera simulación: [MicroDuck RL](/docs/practices/humanoid/microduck-rl) o [ACT bimanual](/docs/practices/vla/act), preparando su entorno.
- Equilibrio de bípedos con ruedas: [avance de Flamingo](/docs/practices/wheel-legged/flamingo-isaaclab/preview), con requisitos y experimentos previstos.

Comprueba entorno y objetivos antes de empezar. Conserva código, parámetros y resultados, y analiza evaluaciones y fallos. Los capítulos de los proyectos están actualmente en chino; los estados siguientes describen ese contenido original.

## Direcciones \{#directions}

| Dirección | Proyectos | Adecuada para |
| --- | --- | --- |
| [Brazos](/docs/practices/robot-arm/placeholder) | 5 | VLA, datos e imitación |
| [Cuadrúpedos](/docs/practices/quadruped/placeholder) | 4 | Refuerzo, control y sim-to-real |
| [Bípedos y humanoides](/docs/practices/humanoid/placeholder) | 1 disponible, 2 previstos | Control avanzado, refuerzo y planificación |
| [Manipulación móvil](/docs/practices/mobile-manipulation/placeholder) | 3 | Navegación y manipulación |
| [Bípedos con ruedas](/docs/practices/wheel-legged/placeholder) | Avance | Equilibrio subactuado, Isaac Lab y validación entre simuladores |

## Proyectos disponibles \{#available-projects}

- [MicroDuck RL: caminar, levantarse y rodar](/docs/practices/humanoid/microduck-rl): MJCF, actuadores BAM y registro de tareas mjlab; 18 modos de movimiento con MuJoCo Warp y evaluación sin pantalla mediante GIF y MP4.

## Proyectos AMD \{#amd}

- [AUP Learning Cloud](/docs/practices/amd/aup-learning-cloud): APU Ryzen AI, JupyterHub, Code Server y ROCm en el navegador para ejercicios, inferencia local y pequeños experimentos.
- [MicroDuck RL · AMD ROCm](/docs/practices/amd/microduck-rl): compilar ROCm Warp y MuJoCo Warp en Radeon R9700, corregir la caché de broadphase dinámico y entrenar PPO bípedo.
- [ACT bimanual · AMD ROCm](/docs/practices/amd/vla-act): BF16 en GPU Radeon, reanudar checkpoints, evaluar 20 episodios y exportar vídeos.
- [Explora Pupper](/docs/practices/amd/pupper-control/intro): proyecto destacado con **Pupper Locomotion** para políticas RL y **Pupper VLA** para visión, lenguaje y acción.

## Proyectos de simulación \{#simulation}

| Proyecto | Ruta técnica | Estado |
| --- | --- | --- |
| [Cuadrúpedo desde cero](/docs/practices/quadruped/cs123/intro) | MuJoCo, PD, cinemática, marchas, políticas y percepción | Resumen y 8 capítulos disponibles |
| [MicroDuck RL](/docs/practices/humanoid/microduck-rl) | mjlab, MuJoCo Warp, PPO paralelo en CUDA y 18 movimientos | Disponible |
| [Primeros pasos con MuJoCo](/docs/practices/robot-arm/mujoco-arm-pick-place) | MJCF, física y control Python | Disponible |
| [DDPG InvertedPendulum](/docs/practices/robot-arm/ddpg-mujoco/invertedpendulum-v5) | Control continuo y referencia DDPG | Disponible |
| [DDPG Reacher](/docs/practices/robot-arm/ddpg-mujoco/reacher-v5) | Seguimiento de objetivos con brazo plano | Disponible |
| [DDPG Pusher](/docs/practices/robot-arm/ddpg-mujoco/pusher-v5) | Contacto y diseño de recompensas | Disponible |
| [ACT bimanual](/docs/practices/vla/act) | ALOHA, imitación y evaluación multiepisodio | Disponible |
| [π₀.₅ + RECAP: LIBERO](/docs/practices/vla/recap) | Modelos de valor, ventajas, ACP y evaluación LIBERO | Disponible |
| [Validación Sim2Sim](/docs/practices/quadruped/sim2sim/placeholder) | Políticas entre simuladores | En desarrollo |
| [Flamingo · Isaac Lab](/docs/practices/wheel-legged/flamingo-isaaclab/preview) | Equilibrio, refuerzo y validación entre simuladores | Avance del curso |

## Proyectos con robots reales \{#real-robots}

LeRobot introduce datos y herramientas; SO-101 aplica lo aprendido a un brazo real.

| Proyecto | Ruta técnica | Estado |
| --- | --- | --- |
| [LeRobot en chino](/docs/practices/robot-arm/data-collection/lerobot-course) | Aprendizaje robótico, datos, herramientas y robótica clásica | Unidades 0–2 recopiladas |
| [SO-101 + LeRobot](/docs/practices/robot-arm/data-collection/so101-lerobot-real) | Conectividad, seguridad y reproducción de acciones | Disponible |
| [Control de brazos ROS2](/docs/practices/robot-arm/ros2-arm-control/placeholder) | Control ROS2 y ejecución | En desarrollo |
| [Guía Sim2Real](/docs/practices/quadruped/sim2real-guide/placeholder) | Despliegue de políticas y validación real | En desarrollo |
