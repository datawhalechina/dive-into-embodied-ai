---
title: "Quaddle: miglioramenti e limiti dopo il passo più rapido"
sidebar_position: 4
displayed_sidebar: practicesQuadrupedSidebar
description: "Riproduci il controllo a 3,34 Hz e distingui frequenza articolare, corsa con fase aerea, terreni superati e limiti del modello."
---

# Quaddle: miglioramenti e limiti dopo il passo più rapido

**Le zampe possono muoversi più frequentemente senza avanzare velocemente.** Lo studio del 16 novembre 2026 ha prodotto un candidato simulato a circa 3,34 Hz. Nelle stesse sei condizioni perturbate, il vecchio controllore ne supera 2/6 e il nuovo 6/6. Non sono risolti il terreno accidentato originale, la salita continua dei gradini o la corsa con fase aerea. I risultati precedenti sono negli [esperimenti multiscenario](./experiments).

È un miglioramento del controllo in retroazione su quattro articolazioni: nessun nuovo addestramento PPO, nessuna sostituzione dell'andatura predefinita, modifica della dinamica o verifica hardware. Ogni prova dura 20 secondi di movimento più 2 di avvio; 6/6 non significa affidabilità statistica del 100%.

## Osservare movimento e direzione \{#cadence}

<video controls playsInline preload="none" src={require('@site/docs/practices/quadruped/quaddle-mujoco/figs/optimization/cadence-optimized.mp4').default} poster={require('@site/docs/practices/quadruped/quaddle-mujoco/figs/optimization/cadence-optimized.webp').default} style={{width: '100%', maxWidth: 960}} />

Il filmato è una riproduzione reale della simulazione di 22 secondi, a velocità normale, incluso l'avvio. Hz indica i cicli completi di andata e ritorno di una singola articolazione motorizzata al secondo, non il totale degli appoggi delle quattro zampe né l'aggiornamento del controllo a 100 Hz.

| Misura nominale | Vecchio candidato a 3,34 Hz | Nuovo candidato |
| --- | ---: | ---: |
| Velocità media in avanti | 0,07025 m/s | 0,06953 m/s |
| Deviazione massima di orientamento | 17,39° | 7,62° |
| Picco reale delle articolazioni motorizzate | 8,929 rad/s | 8,929 rad/s |
| Frequenza principale reale | Circa 3,34 Hz | Circa 3,34 Hz |

Ogni zampa del nuovo candidato presenta 57 picchi angolari evidenti fra i secondi 5 e 22. Il picco FFT è 3,352941 Hz e la stima dagli intervalli fra picchi è circa 3,333333 Hz, coerenti con il comando di 3,342608 Hz alla risoluzione del campionamento. L'escursione angolare reale P95−P5 è 0,416–0,486 rad: il movimento rimane significativo.

### Modifica della correzione direzionale \{#controller}

Frequenza, ampiezze di base e fase fra zampe anteriori e posteriori restano fisse. La correzione usa una piccola differenza degli angoli neutri fra sinistra e destra, eliminando il termine di posizione laterale dall'errore di orientamento. Si scrivono solo quattro obiettivi di posizione, senza modificare lo stato del corpo o applicare spinte esterne.

```text
steering_profile = bias
heading_kp = 0.1
heading_kd = 0.03
steering_limit = 0.12 rad
lateral_kp = 0
frequency = 3.342608 Hz
```

Restano il limite di variazione degli obiettivi di 8 rad/s e il controllo della velocità articolare reale di 10 rad/s. Disattivare soltanto la retroazione laterale produce ancora circa 179,91° di deviazione con attrito ridotto: eliminare un solo termine non spiega tutto. Guadagni maggiori non sono automaticamente migliori: un candidato sulle ampiezze posteriori supera la selezione iniziale ma raggiunge 61,68° e 10,703 rad/s con attrito maggiore.

### Congelare i parametri prima della verifica \{#validation}

Si seleziona su sei condizioni di sviluppo, poi si congela un candidato per le sei prove seguenti. Il vecchio controllore usa esattamente le stesse condizioni. Il nuovo supera tutte e sei le condizioni di sviluppo e tutte e sei le verifiche successive.

| Condizione | Deviazione massima precedente | Deviazione massima nuova | Verifica del passo: vecchio / nuovo |
| --- | ---: | ---: | --- |
| Orientamento −1,5°, attrito ×0,95 | 11,92° | 8,19° | Superata / superata |
| Orientamento +1,5°, attrito ×1,05 | 119,09° | 6,50° | Fallita / superata |
| Posizione laterale −7,5 mm | 16,89° | 7,98° | Superata / superata |
| Posizione laterale +7,5 mm | 52,27° | 6,96° | Fallita / superata |
| Attrito ×0,95 | 169,87° | 6,70° | Fallita / superata |
| Attrito ×1,05 | 179,27° | 7,80° | Fallita / superata |

Nelle nuove prove la velocità è 0,06579–0,07140 m/s e il massimo articolare è 9,090 rad/s. Passano i controlli di postura, contatto, chiusura dei leveraggi, coppia e contabilizzazione delle forze.

![Confronto nelle stesse condizioni di orientamento, velocità e movimento articolare](@site/docs/practices/quadruped/quaddle-mujoco/figs/optimization/cadence-comparison.webp)

È un miglioramento del mantenimento dell'orientamento, non un inseguimento preciso della traiettoria. La deriva laterale arriva ancora a circa 7,2 cm; nella prima condizione cresce da circa 5,7 a 7,2 cm. Non tutte le misure migliorano insieme.

## Obiettivi ancora incompleti \{#bottlenecks}

| Scenario / obiettivo | Progresso documentato | Limite attuale |
| --- | --- | --- |
| Camminata lenta ad alta frequenza | Circa 3,34 Hz; verifica nelle stesse condizioni da 2/6 a 6/6 | Valide solo le prove brevi elencate; a 4–6 Hz le articolazioni seguono il comando ma la direzione è ancora errata |
| Gradino singolo di 2 mm | La retrazione della zampa bloccata risolve un caso con bordo spostato di +15 mm: 85,37°→20,53° | Altre posizioni falliscono; nessuna garanzia generale sotto 2 mm |
| Tre gradini di 2 mm | Passaggio nominale e miglioramenti in alcune perturbazioni | Il recupero congelato supera soltanto 4/10 condizioni miste e peggiora un caso precedentemente superato; non adottato globalmente |
| Terreno originale ±3 mm | Identificati salti adiacenti fino a circa 5,759 mm e problemi di contatto/direzione | Falliscono tutte e tre le mappe originali provate; il terreno attenuato non sostituisce il compito originale |
| Tre gradini di 4 mm in salita | Conservate le prove di fallimento | Spinta e orientamento irrisolti; ottimizzazione sospesa, impossibilità meccanica non dimostrata |
| Corsa originale con fase aerea | Il controllo predittivo nominale combina limite e volo: 0,06011 m/s, 9,938 rad/s, 27 intervalli | Sotto 0,1 m/s; con attrito ridotto può tornare il superamento di velocità; nessuna prova completa superata |
| Camminare→correre→decelerare→girare→fermarsi | Nel primo PPO passano camminata e svolta | Falliscono corsa, decelerazione e arresto; ulteriori ottimizzazioni sospese |

Le precedenti verifiche su piano, pendenze, discesa dei gradini e terreno attenuato mantengono il proprio ambito limitato. **La configurazione ad alta frequenza non è stata verificata su quei terreni e non sostituisce i rispettivi controllori.**

Il PPO dedicato alla corsa ha usato 131.072 passi iniziali e altri 131.072 nei confronti, senza superare il compito. La sequenza combinata ha usato altri 32.768 passi e non è riuscita complessivamente. Il miglioramento recente deriva dal controllo in retroazione, non dall'apprendimento di queste azioni da parte del PPO. Questo PR contiene risultati e riproduzione del passo rapido, non quei checkpoint o l'intera procedura di riaddestramento PPO.

### Passo rapido e corsa hanno criteri distinti \{#criteria}

Il passo richiede comando ≥3,3 Hz, errore della frequenza reale ≤15% per ogni zampa, escursione angolare P95−P5 ≥0,12 rad per zampa, avanzamento ≥0,1 m in 20 secondi e tutti i controlli fisici e direzionali originali. Non richiede fase aerea o 0,1 m/s.

L'originale `run_attempt` richiede ancora almeno 0,1 m/s, almeno tre intervalli completi senza contatto con il terreno di almeno 10 ms ciascuno e gli altri controlli originali. Lo script salva sia `cadence_pass` sia l'originale `result.passed`: il nuovo obiettivo non riscrive un fallimento della corsa.

### Evidenze mancanti sul modello \{#model-limits}

Ogni zampa ha un attuatore e leveraggi passivi: non sono disponibili comandi indipendenti del ginocchio o traiettorie arbitrarie del piede. Le molle sono approssimazioni torsionali non calibrate; anche servo, attrito e contatti non sono calibrati sull'hardware. La soglia di 10 rad/s è un criterio del progetto, non il massimo misurato del motore. I fallimenti non dimostrano un difetto hardware e il successo simulato non garantisce quello reale.

I prossimi studi sono distinti: durata e sequenze di comandi per il passo rapido, appoggi prima del bordo e supporto per i gradini, ottimizzazione congiunta di spinta e velocità per la corsa. Prima del trasferimento reale servono calibrazione del modello e delle interfacce, non soltanto più passi con la stessa ricetta.

## Riprodurre il confronto \{#reproduce}

Installa le dipendenze Python dalla [pagina introduttiva](./) e verifica che le mesh LFS siano file reali. Da `codes/practices/quadruped/quaddle-mujoco`:

```bash
# Configurazioni congelate, caso nominale e sei perturbazioni, 22 s ciascuno
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python evaluate_cadence.py --workers 3 --output outputs/cadence-review
# Registra solo il caso nominale ottimizzato; usa una nuova cartella
MUJOCO_GL=egl .venv/bin/python evaluate_cadence.py --controller optimized --suite nominal --video --output outputs/cadence-video
```

Gli output comprendono controlli originali, risultato separato del passo, campioni e misure di frequenza. I singoli risultati esistenti non vengono sovrascritti. `cadence_controller.py` e `reference/cadence-*.json` sono ingressi sperimentali; i valori predefiniti Python e web restano invariati.

Il file sorgente `reports/optimization-2026-11-16.json` conserva lo studio e le misure di tutte le 73 prove. Ambiente: Python 3.12, MuJoCo 3.16.0; commit sorgente `b0e5ec7`; SHA256 del modello `10d1990f12c2dc79b739b67203422ff8a0090a01e424dacf780b9a03d6015b83`. La simulazione deterministica dei contatti è sensibile a versioni e differenze numeriche: confronta controlli completi e traiettorie.

## Diagnosticare un fallimento \{#exercise}

1. Confronta `cadence_pass` con l'originale `result.passed`: perché una traiettoria può superare il passo e fallire la corsa?
2. In una cartella distinta cambia soltanto orientamento iniziale o attrito. Esamina orientamento, frequenza reale e deriva laterale.
3. Usa gli angoli articolari per controllare se un comando di 4 Hz genera davvero movimento a 4 Hz, poi valuta la direzione. Misura separatamente frequenza comandata, frequenza reale e avanzamento.
