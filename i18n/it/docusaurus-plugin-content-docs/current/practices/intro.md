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
- Equilibrio dei bipedi su ruote e validazione tra simulatori: il [corso Flamingo · Isaac Lab](/docs/practices/humanoid/flamingo-isaaclab/intro) guida in sei capitoli tra PPO parallelo, Constraints as Terminations (CaT), esportazione ONNX e test di robustezza in MuJoCo.

Verifica ambiente e obiettivi prima di iniziare. Conserva codice, parametri e risultati e analizza valutazioni e guasti. I capitoli dei progetti sono attualmente in cinese; gli stati seguenti descrivono quel contenuto originale.

## Direzioni \{#directions}

| Direzione | Progetti | Adatta a |
| --- | --- | --- |
| [Serie bracci robotici](/docs/practices/robot-arm/placeholder) | 11 disponibili | Controllo continuo, VLA, raccolta dati, imitazione, ROS 2 e bracci reali |
| [Serie quadrupedi](/docs/practices/quadruped/placeholder) | 6 disponibili | Apprendimento per rinforzo, cinematica, andature, MJX e Sim2Sim, Sim2Real, navigazione VLA |
| [Serie bipedi](/docs/practices/humanoid/placeholder) | 9 disponibili | Umanoide Unitree G1, Flamingo su Isaac Lab + CaT, PPO parallelo su GPU e curriculum |
| [Manipolazione mobile](/docs/practices/mobile-manipulation/placeholder) | 3 disponibili | Navigazione SLAM + Nav2, navigazione visione-linguaggio VLN e coordinazione base + braccio |

## Progetti disponibili \{#available-projects}

- [Locomozione umanoide Unitree G1 (4 parti)](/docs/practices/humanoid/unitree-g1/): analisi delle 16 ricompense e delle curve PPO su 10.000 iterazioni, Sim2Sim MuJoCo a 0,2–0,8 m/s, esportazione LSTM ONNX nel browser (provabile in [Simulazione](/simulation)) e approfondimenti su AMP, Get-Up e GR00T Whole-Body Control.
- [Locomozione quadrupede Go2 · MJX (ROCm / CUDA)](/docs/practices/amd/unitree-rl-mjx): porting in MuJoCo MJX / JAX di `unitree_rl_mjlab`, con lo stesso codice di addestramento su AMD ROCm e NVIDIA CUDA ed esportazione ONNX nello stack C++ Sim2Sim ufficiale.
- [Flamingo · Isaac Lab](/docs/practices/humanoid/flamingo-isaaclab/intro): pipeline completa per un bipede su ruote a 8 gradi di libertà, con attuatori ritardati, PPO vs CaT su 4096 ambienti, fusione ONNX e test zero-shot in MuJoCo (panoramica e 6 capitoli).
- [MicroDuck RL: camminare, rialzarsi e rotolare](/docs/practices/humanoid/microduck-rl): MJCF, attuatori BAM e registrazione dei compiti mjlab; 18 modalità di movimento con MuJoCo Warp e valutazione senza schermo tramite GIF e MP4.

## Progetti AMD \{#amd}

- [AUP Learning Cloud](/docs/practices/amd/aup-learning-cloud): APU Ryzen AI, JupyterHub, Code Server e ROCm nel browser per esercizi, inferenza locale e piccoli esperimenti.
- [Locomozione Go2 · MJX (ROCm / CUDA)](/docs/practices/amd/unitree-rl-mjx): addestramento di politiche di velocità Unitree Go2 su GPU da 8 GB con ROCm e CUDA, vincoli hardware RDNA4, 980 milioni di passi e verifica Sim2Sim nello stack ufficiale.
- [MicroDuck RL · AMD ROCm](/docs/practices/amd/microduck-rl): compilare ROCm Warp e MuJoCo Warp su Radeon R9700, correggere la cache del broadphase dinamico e addestrare PPO bipede.
- [ACT bimanuale · AMD ROCm](/docs/practices/amd/vla-act): BF16 su GPU Radeon, ripresa dei checkpoint, valutazione di 20 episodi ed esportazione video.
- [Esplora Pupper](/docs/practices/amd/pupper-control/intro): progetto di punta con **Pupper Locomotion** per politiche RL e **Pupper VLA** per visione, linguaggio e azione.

## Progetti di simulazione \{#simulation}

| Progetto | Percorso tecnico | Stato |
| --- | --- | --- |
| [Quadrupede da zero](/docs/practices/quadruped/cs123/intro) | MuJoCo, PD, cinematica, andature, politiche e percezione | Panoramica e 8 capitoli disponibili |
| [Locomozione quadrupede Go2 · MJX (ROCm / CUDA)](/docs/practices/amd/unitree-rl-mjx) | MuJoCo MJX / JAX, addestramento multipiattaforma, ONNX e Sim2Sim ufficiale | Panoramica e 3 tutorial disponibili |
| [Simulazione Petoi Quaddle con quattro servomotori](/docs/practices/quadruped/quaddle-mujoco) | Adattamento dell'URDF ufficiale, controllo dal browser, andature periodiche e addestramento PPO residuo | Disponibile |
| [MicroDuck RL](/docs/practices/humanoid/microduck-rl) | mjlab, MuJoCo Warp, PPO parallelo su CUDA e 18 movimenti | Disponibile |
| [Primi passi con MuJoCo](/docs/practices/robot-arm/mujoco-arm-pick-place) | MJCF, fisica e controllo Python | Disponibile |
| [DDPG InvertedPendulum](/docs/practices/robot-arm/ddpg-mujoco/invertedpendulum-v5) | Controllo continuo e riferimento DDPG | Disponibile |
| [DDPG Reacher](/docs/practices/robot-arm/ddpg-mujoco/reacher-v5) | Inseguimento con braccio planare | Disponibile |
| [DDPG Pusher](/docs/practices/robot-arm/ddpg-mujoco/pusher-v5) | Contatto e progettazione delle ricompense | Disponibile |
| [ACT bimanuale](/docs/practices/vla/act) | ALOHA, imitazione e valutazione su più episodi | Disponibile |
| [π₀.₅ + RECAP: LIBERO](/docs/practices/vla/recap) | Modelli di valore, vantaggi, ACP e valutazione LIBERO | Disponibile |
| [Navigazione VLA con Petoi Quaddle](/docs/practices/vla/quaddle/) | Fine-tuning di SmolVLA, controllo CPG, temporal ensembling e decomposizione dei compiti | Disponibile |
| [Guida alla validazione Sim2Sim](/docs/practices/quadruped/sim2sim/placeholder) | Allineamento tra simulatori, checklist diagnostica e casi Go2 / Flamingo | Disponibile |
| [Locomozione umanoide Unitree G1](/docs/practices/humanoid/unitree-g1/) | Analisi 16 ricompense, 10k iterazioni PPO, Sim2Sim multi-velocità, LSTM nel browser e GR00T WBC | Panoramica e 3 guide disponibili |
| [Flamingo · Isaac Lab](/docs/practices/humanoid/flamingo-isaaclab/intro) | Equilibrio a 8 DoF, PPO vs CaT, esportazione ONNX e test Sim2Sim in MuJoCo | Panoramica e 6 capitoli disponibili |
| [Equilibrio bipede e recupero da spinte](/docs/practices/humanoid/balance-control/placeholder) | Controllo CoM/momento angolare, test di spinta e simulazione Unitree G1 nel browser | Disponibile |
| [Motion tracking umanoide full-body](/docs/practices/humanoid/motion-tracking/placeholder) | Retargeting mocap, ricompense di tracciamento e imitazione a corpo intero | Disponibile |
| [Pianificazione di alto livello per bipedi](/docs/practices/humanoid/task-planning-demo/placeholder) | Decomposizione VLM/LLM, orchestrazione abilità e vincoli di sicurezza | Disponibile |
| [Serie manipolazione mobile (Nav / VLN / Coop)](/docs/practices/mobile-manipulation/placeholder) | Navigazione SLAM + Nav2, navigazione visione-linguaggio e manipolazione mobile coordinata | 3 guide disponibili |

## Progetti con robot reali \{#real-robots}

LeRobot introduce dati e strumenti; SO-101 applica quanto appreso a un braccio reale.

| Progetto | Percorso tecnico | Stato |
| --- | --- | --- |
| [LeRobot in cinese](/docs/practices/robot-arm/data-collection/lerobot-course) | Apprendimento robotico, dati, strumenti e robotica classica | Unità 0–2 raccolte |
| [SO-101 + LeRobot](/docs/practices/robot-arm/data-collection/so101-lerobot-real) | Connettività, sicurezza e riproduzione delle azioni | Disponibile |
| [Controllo bracci con ROS 2](/docs/practices/robot-arm/ros2-arm-control/placeholder) | `ros2_control`, pianificazione MoveIt 2 ed esecuzione sicura | Disponibile |
| [Imitation learning per compiti personalizzati](/docs/practices/robot-arm/imitation-learning/placeholder) | Raccolta teleoperata, cura dei dati e ciclo ACT / Diffusion Policy | Disponibile |
| [Guida ingegneristica Sim2Real](/docs/practices/quadruped/sim2real-guide/placeholder) | Identificazione di sistema, domain randomization, sicurezza e messa in servizio | Disponibile |
