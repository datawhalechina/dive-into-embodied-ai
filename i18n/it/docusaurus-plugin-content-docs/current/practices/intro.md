---
title: "Progetti: dalla simulazione ai robot reali"
description: "Scegli progetti per piattaforma, simulazione e impiego su robot reali."
sidebar_position: 1
displayed_sidebar: practicesOverviewSidebar
---
# Progetti: dalla simulazione ai robot reali

Scegli un compito e affronta ambiente, addestramento, controllo e valutazione con un progetto, una demo o una riproduzione. Segui capitoli per costruire un sistema o un esperimento autonomo per provare un metodo. Consulta i [fondamenti](/docs/foundations/intro) per i principi e le [risorse](/docs/introduction/resources) per articoli, dati e codice.

## Da dove iniziare \{#getting-started}

- Da zero: il [progetto sui quadrupedi](/docs/practices/quadruped/cs123/intro) ha otto capitoli su attuatori, cinematica, modellazione, andature e politiche. Serve Python di base; puoi imparare controllo e algebra lineare durante la pratica.
- Dati e hardware dei bracci: [dispense LeRobot in cinese](/docs/practices/robot-arm/data-collection/lerobot-course), unità 0–2 e Python di base, poi [SO-101](/docs/practices/robot-arm/data-collection/so101-lerobot-real), che richiede un robot compatibile.
- Prima simulazione: [MicroDuck RL](/docs/practices/humanoid/microduck-rl) o [ACT bimanuale](/docs/practices/vla/act), configurando l'ambiente.
- Equilibrio dei bipedi su ruote: [anteprima Flamingo](/docs/practices/wheel-legged/flamingo-isaaclab/preview), con prerequisiti ed esperimenti previsti.

Verifica ambiente e obiettivi prima di iniziare. Conserva codice, parametri e risultati e analizza valutazioni e guasti. I capitoli dei progetti sono attualmente in cinese; gli stati seguenti descrivono quel contenuto originale.

## Direzioni \{#directions}

| Direzione | Progetti | Adatta a |
| --- | --- | --- |
| [Bracci](/docs/practices/robot-arm/placeholder) | 5 | VLA, dati e imitazione |
| [Quadrupedi](/docs/practices/quadruped/placeholder) | 4 | Rinforzo, controllo e sim-to-real |
| [Bipedi e umanoidi](/docs/practices/humanoid/placeholder) | 1 disponibile, 2 previsti | Controllo avanzato, rinforzo e pianificazione |
| [Manipolazione mobile](/docs/practices/mobile-manipulation/placeholder) | 3 | Navigazione e manipolazione |
| [Bipedi su ruote](/docs/practices/wheel-legged/placeholder) | Anteprima | Equilibrio sottoattuato, Isaac Lab e validazione tra simulatori |

## Progetti disponibili \{#available-projects}

- [MicroDuck RL: camminare, rialzarsi e rotolare](/docs/practices/humanoid/microduck-rl): MJCF, attuatori BAM e registrazione dei compiti mjlab; 18 modalità di movimento con MuJoCo Warp e valutazione senza schermo tramite GIF e MP4.

## Progetti AMD \{#amd}

- [AUP Learning Cloud](/docs/practices/amd/aup-learning-cloud): APU Ryzen AI, JupyterHub, Code Server e ROCm nel browser per esercizi, inferenza locale e piccoli esperimenti.
- [MicroDuck RL · AMD ROCm](/docs/practices/amd/microduck-rl): compilare ROCm Warp e MuJoCo Warp su Radeon R9700, correggere la cache del broadphase dinamico e addestrare PPO bipede.
- [ACT bimanuale · AMD ROCm](/docs/practices/amd/vla-act): BF16 su GPU Radeon, ripresa dei checkpoint, valutazione di 20 episodi ed esportazione video.
- [Esplora Pupper](/docs/practices/amd/pupper-control/intro): progetto di punta con **Pupper Locomotion** per politiche RL e **Pupper VLA** per visione, linguaggio e azione.

## Progetti di simulazione \{#simulation}

| Progetto | Percorso tecnico | Stato |
| --- | --- | --- |
| [Quadrupede da zero](/docs/practices/quadruped/cs123/intro) | MuJoCo, PD, cinematica, andature, politiche e percezione | Panoramica e 8 capitoli disponibili |
| [MicroDuck RL](/docs/practices/humanoid/microduck-rl) | mjlab, MuJoCo Warp, PPO parallelo su CUDA e 18 movimenti | Disponibile |
| [Primi passi con MuJoCo](/docs/practices/robot-arm/mujoco-arm-pick-place) | MJCF, fisica e controllo Python | Disponibile |
| [DDPG InvertedPendulum](/docs/practices/robot-arm/ddpg-mujoco/invertedpendulum-v5) | Controllo continuo e riferimento DDPG | Disponibile |
| [DDPG Reacher](/docs/practices/robot-arm/ddpg-mujoco/reacher-v5) | Inseguimento con braccio planare | Disponibile |
| [DDPG Pusher](/docs/practices/robot-arm/ddpg-mujoco/pusher-v5) | Contatto e progettazione delle ricompense | Disponibile |
| [ACT bimanuale](/docs/practices/vla/act) | ALOHA, imitazione e valutazione su più episodi | Disponibile |
| [π₀.₅ + RECAP: LIBERO](/docs/practices/vla/recap) | Modelli di valore, vantaggi, ACP e valutazione LIBERO | Disponibile |
| [Validazione Sim2Sim](/docs/practices/quadruped/sim2sim/placeholder) | Politiche tra simulatori | In sviluppo |
| [Flamingo · Isaac Lab](/docs/practices/wheel-legged/flamingo-isaaclab/preview) | Equilibrio, rinforzo e validazione tra simulatori | Anteprima del corso |

## Progetti con robot reali \{#real-robots}

LeRobot introduce dati e strumenti; SO-101 applica quanto appreso a un braccio reale.

| Progetto | Percorso tecnico | Stato |
| --- | --- | --- |
| [LeRobot in cinese](/docs/practices/robot-arm/data-collection/lerobot-course) | Apprendimento robotico, dati, strumenti e robotica classica | Unità 0–2 raccolte |
| [SO-101 + LeRobot](/docs/practices/robot-arm/data-collection/so101-lerobot-real) | Connettività, sicurezza e riproduzione delle azioni | Disponibile |
| [Controllo bracci ROS2](/docs/practices/robot-arm/ros2-arm-control/placeholder) | Controllo ROS2 ed esecuzione | In sviluppo |
| [Guida Sim2Real](/docs/practices/quadruped/sim2real-guide/placeholder) | Impiego di politiche e validazione reale | In sviluppo |
