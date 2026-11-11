# Libreria di simulazione online

[中文](README.md) · [English](README.en.md)

La pagina `/simulation` condivide un catalogo tra cinese, inglese e italiano. **Controllo libero** converte tastiera e pulsanti tattili in comandi. **Dimostrazione** invia automaticamente il `demoCommand` del modello. Fisica e inferenza vengono eseguite in un Worker del browser, senza un servizio remoto di addestramento.

## Modelli inclusi

| Modello | Controllore | Movimento |
| --- | --- | --- |
| Unitree Go2 | Pesi ONNX PPO v0 del repository, inferenza a 50 Hz | Avanti, indietro, rotazione e traslazione laterale |
| Petoi Quaddle | Andatura periodica esistente, senza pesi addestrati | Avanti, curve camminando, arresto |

Go2 usa 47 osservazioni nell'ordine FL / FR / RL / RR: velocità angolare locale, gravità proiettata, comandi, fase, stato articolare e azione precedente. La normalizzazione è nel grafo ONNX. Le azioni diventano posizioni obiettivo e coppie PD, associate agli attuatori per nome. Stop invia velocità zero; pausa congela la fisica.

Il modello `scene_go2.xml` conserva inerzie, giunti, attuatori e collisioni. La compilazione rimuove le mesh visive che richiedono un download separato e mostra le forme di collisione. Provenienza, hash dei pesi e licenze accompagnano le risorse. I controlli nel browser non verificano hardware reale o qualità dell'addestramento.

## Aggiungere modelli addestrati

1. Conserva risorse distribuibili, pesi, configurazione di inferenza e licenze nel progetto pratico. Esporta ONNX compatibile con il browser; `.pt`, `.pth` e archivi SB3 `.zip` non sono eseguibili direttamente.
2. Registra ID, nome, `kind`, `adapter`, percorsi, corpo seguito, distanza della telecamera, comando dimostrativo e tutorial in `models.json`. Altre politiche con lo stesso contratto Go2 possono riutilizzare l'adattatore.
3. Per robot o politiche diversi, aggiungi un adattatore in `runtime/` e collegalo a `worker.mjs` e `scripts/simulation-assets.cjs`. Definisci ordine e coordinate delle osservazioni, cronologia, normalizzazione, scala delle azioni, ordine dei giunti, frequenza, inizializzazione e ripristino. ONNX da solo non basta. L'adattatore attuale supporta il contratto Go2, non pesi caricati arbitrariamente.
4. Aggiorna le descrizioni nelle tre lingue. Modifica solo i sorgenti: `static/simulation-assets/` e `static/quaddle/` sono generati. Non scaricare modelli senza versione fissata e non eseguire script degli utenti durante la compilazione.
5. Aggiungi verifiche fisiche e di inferenza per movimento, arresto, ripristino e output non validi. Esegui con npm: `check:simulation`, `check:quaddle-web`, `typecheck`, `assets:webp:check`, `build`, `check:i18n`, `check:page-markdown`.

Nel browser verifica tastiera e tocco, cambio modalità/modello, rilascio, perdita del focus, pagina nascosta, pausa/ripristino, caricamenti ripetuti, errori, schermi stretti e temi chiaro/scuro. Le scorciatoie funzionano solo nella scena attiva; uscendo i comandi vengono azzerati. I dispositivi lenti possono simulare sotto il tempo reale, senza recuperare il tempo trascorso in background.
