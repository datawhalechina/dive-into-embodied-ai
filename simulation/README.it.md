# Libreria di simulazione online

[中文](README.md) · [English](README.en.md)

La pagina `/simulation` condivide un catalogo tra cinese, inglese e italiano. **Controllo libero** converte tastiera e pulsanti tattili in comandi. **Dimostrazione** invia automaticamente il `demoCommand` del modello. Fisica e inferenza vengono eseguite in un Worker del browser, senza un servizio remoto di addestramento.

## Modelli inclusi

| Modello | Controllore | Movimento |
| --- | --- | --- |
| Unitree Go2 | Pesi ONNX PPO v0 del repository, inferenza a 50 Hz | Avanti, indietro, rotazione e traslazione laterale |
| Pupper v3 | Checkpoint PPO a 8 milioni di passi, inferenza 50 Hz | Avanti, indietro, rotazione, traslazione |
| MicroDuck | 14 servi di posizione, senza pesi addestrati | Tasti testa, cursori articolari, base fissa/libera |
| Petoi Quaddle | Andatura periodica esistente, senza pesi addestrati | Avanti, curve camminando, arresto |

Go2 usa 47 osservazioni nell'ordine FL / FR / RL / RR: velocità angolare locale, gravità proiettata, comandi, fase, stato articolare e azione precedente. La normalizzazione è nel grafo ONNX. Le azioni diventano posizioni obiettivo e coppie PD, associate agli attuatori per nome. Stop invia velocità zero; pausa congela la fisica.

Il modello `scene_go2.xml` conserva inerzie, giunti, attuatori e collisioni. La compilazione rimuove le mesh visive che richiedono un download separato e mostra le forme di collisione. Provenienza, hash dei pesi e licenze accompagnano le risorse. I controlli nel browser non verificano hardware reale o qualità dell'addestramento.

## Pupper v3 e MicroDuck

Pupper v3 usa il checkpoint PPO a 8 milioni di passi di `cs123/6.rl_pupper_v2_5`, con mesh originali. Le 53 osservazioni seguono FR / FL / RR / RL, incluse trasformazioni dell'ambiente di addestramento e fase trot. Passo fisico 4 ms, controllo 20 ms, scala azioni 0,5, filtro EMA, KP=5 / KD=0,25. Un comando zero può comunque produrre deriva.

La cartella `web/` del progetto contiene ONNX, configurazione, campioni nativi e hash delle fonti. Per rigenerare, installa `onnx==1.23.2` e `onnxruntime==1.31.1` nell'ambiente cs123 con SB3/PyTorch, poi esegui dalla radice `codes/practices/quadruped/cs123/.venv/bin/python scripts/export-pupper-web.py`. L'esportazione confronta 256 azioni deterministiche SB3/ONNX. La compilazione web usa i file salvati, senza Python o dipendenze di addestramento.

MicroDuck usa mesh originali, 14 servi di posizione e posa STAND. Senza pesi addestrati è disponibile solo il controllo libero: W/S beccheggio, A/D rotazione, Q/E rollio della testa; i cursori regolano tutte le articolazioni. Un vincolo fissa la base. Disattivalo per la fisica libera; riattivarlo o ripristinare riporta alla posa iniziale. Si usano i servi MJCF, non il modello BAM dell'addestramento. Licenza del codice e avviso sulle mesh originali accompagnano le risorse.

## Aggiungere modelli addestrati

1. Conserva risorse distribuibili, pesi, configurazione di inferenza e licenze nel progetto pratico. Esporta ONNX compatibile con il browser; `.pt`, `.pth` e archivi SB3 `.zip` non sono eseguibili direttamente.
2. Registra ID, nome, `kind`, `adapter`, percorsi, corpo seguito, distanza della telecamera, comando dimostrativo e tutorial in `models.json`. Altre politiche con lo stesso contratto Go2 possono riutilizzare l'adattatore.
3. Per robot o politiche diversi, aggiungi un adattatore in `runtime/` e collegalo a `worker.mjs` e `scripts/simulation-assets.cjs`. Definisci ordine e coordinate delle osservazioni, cronologia, normalizzazione, scala delle azioni, ordine dei giunti, frequenza, inizializzazione e ripristino. ONNX da solo non basta. Gli adattatori supportano i contratti noti Go2 e Pupper, non pesi caricati arbitrariamente.
4. Aggiorna le descrizioni nelle tre lingue. Modifica solo i sorgenti: `static/simulation-assets/` e `static/quaddle/` sono generati. Non scaricare modelli senza versione fissata e non eseguire script degli utenti durante la compilazione.
5. Aggiungi verifiche fisiche e di inferenza per movimento, arresto, ripristino e output non validi. Esegui con npm: `check:simulation`, `check:quaddle-web`, `typecheck`, `assets:webp:check`, `build`, `check:i18n`, `check:page-markdown`.

Nel browser verifica tastiera e tocco, cambio modalità/modello, rilascio, perdita del focus, pagina nascosta, pausa/ripristino, caricamenti ripetuti, errori, schermi stretti e temi chiaro/scuro. Le scorciatoie funzionano solo nella scena attiva; uscendo i comandi vengono azzerati. I dispositivi lenti possono simulare sotto il tempo reale, senza recuperare il tempo trascorso in background.
