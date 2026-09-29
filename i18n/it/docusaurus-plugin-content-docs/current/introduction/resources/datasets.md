---
title: "Dataset di IA incarnata"
description: "Dimostrazioni robotiche e benchmark di valutazione con tipi di dati, utilizzi e download ufficiali."
sidebar_position: 11
displayed_sidebar: introductionSidebar
---
# Dataset di IA incarnata

Prima di scegliere dati per addestramento o valutazione, verifica corpo del robot, definizioni di osservazioni e azioni e ambito dei compiti.

## Dati di robot reali \{#real-robot-data}

### Open X-Embodiment (OXE) \{#open-x-embodiment}

Riunisce traiettorie di manipolazione di numerose piattaforme in un formato standardizzato. È utile per studiare miscele di dati tra robot e addestramento di politiche generaliste.

- Tipo: traiettorie reali; osservazioni, azioni e linguaggio variano tra sottoinsiemi.
- Accesso: [progetto e catalogo ufficiale](https://robotics-transformer-x.github.io/), tramite il link Data.
- Verifica: documentazione dei sottoinsiemi, spazio delle azioni, frequenza, dimensioni e condizioni d'uso.
- Studia: [Open X-Embodiment e scala dei dati](/docs/foundations/vla/openx_data_scaling).

## Simulazione e benchmark \{#simulation-benchmarks}

### LIBERO \{#libero}

Organizza compiti di manipolazione per apprendimento continuo e trasferimento, con ambienti, suite e dimostrazioni per addestrare e confrontare politiche.

- Tipo: compiti simulati e traiettorie dimostrate.
- Accesso: [codice, download e compiti ufficiali](https://github.com/Lifelong-Robot-Learning/LIBERO).
- Verifica: suite, partizioni di addestramento e test, versioni dell'ambiente, politiche e osservazioni di valutazione.

## Cosa verificare nella documentazione \{#reading-dataset-docs}

| Aspetto | Cosa confermare |
| --- | --- |
| Robot e compito | Un braccio, due o un altro corpo; presa, manipolazione, locomozione o altro |
| Osservazioni | Viste delle telecamere, immagini, articolazioni, istruzioni e allineamento temporale |
| Azioni | Articolazioni o pose dell'effettore, controllo assoluto o incrementale, unità e frequenza |
| Addestramento e valutazione | Partizioni, traiettorie ripetute, protocolli e condizioni di successo |
| Accesso e uso | Formato, dimensioni, download, licenza e requisiti di citazione |

Per leggere i dati e usare gli strumenti, consulta le [dispense LeRobot in cinese](/docs/practices/robot-arm/data-collection/lerobot-course).

Raccolta e verifica delle fonti: 2026-09-11.
