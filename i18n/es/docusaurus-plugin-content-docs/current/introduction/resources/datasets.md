---
title: "Conjuntos de datos de IA corporizada"
description: "Demostraciones robóticas y benchmarks de evaluación con tipos de datos, usos y descargas oficiales."
sidebar_position: 11
displayed_sidebar: introductionSidebar
---
# Conjuntos de datos de IA corporizada

Antes de elegir datos para entrenamiento o evaluación, comprueba el cuerpo del robot, las definiciones de observaciones y acciones y el alcance de las tareas.

## Datos de robots reales \{#real-robot-data}

### Open X-Embodiment (OXE) \{#open-x-embodiment}

Reúne trayectorias de manipulación de numerosas plataformas en un formato estandarizado. Sirve para estudiar mezcla de datos entre robots y entrenamiento de políticas generalistas.

- Tipo: trayectorias reales; observaciones, acciones y lenguaje varían entre subconjuntos.
- Acceso: [proyecto y catálogo oficial](https://robotics-transformer-x.github.io/), mediante el enlace Data.
- Comprueba: documentación de cada subconjunto, espacio de acciones, frecuencia, tamaño y condiciones de uso.
- Aprende: [Open X-Embodiment y escalado de datos](/docs/foundations/vla/openx_data_scaling).

## Simulación y benchmarks \{#simulation-benchmarks}

### LIBERO \{#libero}

Organiza tareas de manipulación para aprendizaje continuo y transferencia, con entornos, suites y demostraciones para entrenar y comparar políticas.

- Tipo: tareas simuladas y trayectorias de demostración.
- Acceso: [código, descargas y tareas oficiales](https://github.com/Lifelong-Robot-Learning/LIBERO).
- Comprueba: suites, particiones de entrenamiento y prueba, versiones del entorno, políticas y observaciones de evaluación.

## Qué revisar en la documentación \{#reading-dataset-docs}

| Aspecto | Qué confirmar |
| --- | --- |
| Robot y tarea | Un brazo, dos u otro cuerpo; agarre, manipulación, locomoción u otra tarea |
| Observaciones | Vistas de cámara, imágenes, articulaciones, instrucciones y alineación temporal |
| Acciones | Articulaciones o poses del efector, control absoluto o incremental, unidades y frecuencia |
| Entrenamiento y evaluación | Particiones, trayectorias repetidas, protocolos y condiciones de éxito |
| Acceso y uso | Formato, tamaño, descarga, licencia y requisitos de cita |

Para leer datos y usar las herramientas, consulta los [apuntes LeRobot en chino](/docs/practices/robot-arm/data-collection/lerobot-course).

Recopilación y comprobación de fuentes: 2026-09-11.
