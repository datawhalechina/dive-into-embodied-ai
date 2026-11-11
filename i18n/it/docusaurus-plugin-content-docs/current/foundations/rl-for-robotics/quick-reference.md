---
title: Guida rapida a termini e formule
sidebar_position: 1.5
displayed_sidebar: foundationsRlForRoboticsSidebar
description: Simboli, termini e formule comuni dell'apprendimento per rinforzo spiegati con esempi robotici.
---

# Guida rapida a termini e formule

Quando leggi una formula, chiediti prima cosa calcola, poi identifica ogni simbolo e infine prova con numeri piccoli. Questa pagina spiega concetti comuni dei fondamenti di RL. I capitoli su [MDP](/docs/foundations/rl-for-robotics/mdp) e [differenza temporale](/docs/foundations/rl-for-robotics/temporal-difference) includono spiegazioni espandibili accanto alle formule principali. Segui il link di un termine e usa Indietro nel browser per riprendere la lettura. Questa guida è in italiano; i capitoli collegati possono essere ancora in cinese, come indicato dall'avviso nella pagina.

## Come leggere i simboli \{#notation}

| Simbolo | Come leggerlo | Piccolo esempio |
| --- | --- | --- |
| $s_t,a_t,R_{t+1}$ | Stato attuale, azione attuale, ricompensa ricevuta dopo | Un robot arriva a un incrocio, gira a destra e riceve ricompensa $1$ |
| $\sum$ | Somma i termini nell'intervallo indicato | $\sum_{k=0}^{2}x_k=x_0+x_1+x_2$ |
| $\gamma^k$ | Peso di sconto dopo $k$ passi | $0.9^2=0.81$ |
| $\mathbb{E}[X]$ | Media degli esiti pesata con le loro probabilità | Esiti $2,8$ equiprobabili hanno valore atteso $5$ |
| $P(a\mid s)$ | Probabilità dell'azione $a$, dato lo stato $s$ | La barra significa “dato”, non divisione |
| $\leftarrow$ | Sostituisci il valore a sinistra con il risultato a destra | $V\leftarrow V+0.1$ aumenta $V$ di $0.1$ |
| $\max$ / $\arg\max$ | Punteggio massimo / opzione che lo ottiene | Sinistra vale $2$, destra $5$: massimo $5$, opzione migliore destra |
| $\hat{A},\theta$ | Una stima, parametri del modello | Il cappello indica una stima; $\theta$ può indicare i pesi della rete della politica |

La stessa lettera può cambiare significato tra capitoli. Negli MDP, $R_{t+1}$ è una ricompensa; in PPO, $r_t(\theta)$ è un rapporto di probabilità tra politica nuova e precedente. Leggi prima la definizione accanto alla formula.

## Stati, osservazioni e MDP \{#mdp}

Lo **stato (state)** contiene le informazioni necessarie alla previsione; l'**osservazione (observation)** contiene ciò che i sensori misurano. Una foto mostra la posizione di una palla, ma non necessariamente la direzione in cui rotola. Anche la velocità può far parte dello stato.

Un **MDP (processo decisionale di Markov)** organizza un compito in stati $S$, azioni $A$, regole di transizione $P$, regole di ricompensa $R$ e fattore di sconto $\gamma$. È un modello del problema, non una particolare rete neurale: quali informazioni sono disponibili, cosa si può fare, cosa succede dopo e cosa conta come un buon risultato?

La **proprietà di Markov** richiede che, dati stato e azione attuali, la storia passata non aggiunga informazioni necessarie a prevedere lo stato successivo. Un **POMDP** descrive l'osservabilità parziale, per esempio con informazioni nascoste. Aggiungere fotogrammi precedenti può migliorare l'osservazione, ma non garantisce di ricostruire lo stato completo.

Torna al capitolo: [definizione di MDP](/docs/foundations/rl-for-robotics/mdp#21-智能体与环境交互), [proprietà di Markov](/docs/foundations/rl-for-robotics/mdp#22-马尔可夫性质).

## Ricompense, ritorni e sconto \{#return}

La **ricompensa (reward)** è il feedback di un passo; il **ritorno (return)** è la somma delle ricompense future dal momento attuale, eventualmente scontate. Un robot può ricevere $1$ per ogni passo avanti e $10$ all'arrivo. La ricompensa di un passo e il ritorno dell'intero percorso sono quantità diverse.

Il **fattore di sconto (discount factor)** $\gamma$ determina quanto contano oggi le ricompense future. Supponiamo che le prossime due ricompense siano $2,3$, poi l'episodio termini, con $\gamma=0.9$:

$$
G_t=2+0.9\times3=4.7.
$$

Il ritorno dal passo successivo è $G_{t+1}=3$, quindi possiamo anche scrivere $G_t=2+0.9G_{t+1}$. Dopo la terminazione non ci sono ricompense e poniamo $G_T=0$. Nei compiti a orizzonte infinito si usa comunemente $0\leq\gamma<1$ con ricompense limitate per garantire un ritorno finito; negli episodi finiti si può usare anche $\gamma=1$.

Torna al capitolo: [ritorni e forma ricorsiva](/docs/foundations/rl-for-robotics/mdp#24-目标与回报).

## Politiche, valori attesi e funzioni di valore \{#value}

Una **politica (policy)** $\pi$ è una regola per scegliere le azioni. Con azioni discrete, $\pi(a\mid s)=0.7$ significa una probabilità del $70\%$ di scegliere $a$ nello stato $s$. Le azioni continue sono solitamente descritte da densità di probabilità: la densità in un punto non è la probabilità di scegliere esattamente quell'azione.

Il **valore atteso (expectation)** è una media pesata con le probabilità, non un risultato desiderato o garantito in una singola esecuzione.

- **Valore dello stato** $V_\pi(s)$: partendo da qui e seguendo $\pi$, quale ritorno otteniamo in media?
- **Valore dell'azione** $Q_\pi(s,a)$: partendo da qui, eseguendo prima $a$ e poi seguendo $\pi$, quale ritorno otteniamo in media?

Se sinistra e destra hanno valori $Q$ pari a $2,6$ e la politica sceglie ciascuna con probabilità un mezzo, allora $V_\pi(s)=0.5\times2+0.5\times6=4$. È una media secondo la politica, non il massimo $6$.

Torna al capitolo: [politiche e valori](/docs/foundations/rl-for-robotics/mdp#25-策略与价值).

## Equazione di Bellman \{#bellman}

L'**equazione di Bellman** divide il totale futuro medio in “un passo immediato + la media del futuro rimanente”:

$$
V_\pi(s)=\mathbb{E}_\pi[R_{t+1}+\gamma V_\pi(S_{t+1})\mid S_t=s].
$$

Per una politica fissa, si fa la media sia sulle possibili azioni sia sui possibili stati successivi, usando le loro probabilità. L'equazione collega valori attesi; una singola traiettoria che finisce in una buca può comunque ottenere molto meno della media.

Torna al capitolo: [equazioni di Bellman](/docs/foundations/rl-for-robotics/dynamic-programming#32-贝尔曼方程), [da Bellman all'obiettivo TD](/docs/foundations/rl-for-robotics/temporal-difference#53-时序差分目标的推导).

## Bootstrapping, obiettivi TD e tasso di apprendimento \{#td}

In questo contesto, **bootstrapping** significa aggiornare una stima usando un'altra stima. Prima di completare un percorso, usi la stima per la prossima tappa per migliorare quella della tappa attuale. È diverso dal ricampionamento bootstrap in statistica.

L'**obiettivo TD** è la ricompensa osservata più la stima scontata dello stato successivo. L'**errore TD** è questo obiettivo meno la stima attuale. Il **tasso di apprendimento** $\alpha$ determina quanto avvicinarsi all'obiettivo:

$$
\text{nuova stima}=\text{vecchia stima}+\alpha(\text{obiettivo}-\text{vecchia stima}).
$$

Con una vecchia stima di $4$, un obiettivo di $6.4$ e $\alpha=0.1$, la nuova stima è $4.24$. Il tasso di apprendimento e lo sconto sono diversi: $\alpha$ controlla l'aggiornamento, $\gamma$ pesa le ricompense future. Non si applica bootstrapping dopo una vera terminazione; in caso di interruzione per limite di tempo, la scelta dipende dalla definizione di terminazione del compito.

Torna al capitolo: [calcolo di un aggiornamento TD](/docs/foundations/rl-for-robotics/temporal-difference#51-时序差分估计).

## On-policy e off-policy \{#on-off-policy}

La **politica di comportamento (behavior policy)** raccoglie i dati. La **politica obiettivo (target policy)** viene valutata o migliorata. Se coincidono, l'apprendimento è **on-policy**; se differiscono, è **off-policy**.

Sarsa usa il valore dell'azione effettivamente scelta al passo successivo. Q-learning usa il massimo valore stimato delle azioni successive, anche se l'esplorazione ne ha scelta un'altra. La distinzione riguarda quale comportamento si apprende, non l'uso di reti neurali.

Torna al capitolo: [Q-learning e Sarsa](/docs/foundations/rl-for-robotics/temporal-difference#57-q-learning-算法), [on-policy e off-policy](/docs/foundations/rl-for-robotics/temporal-difference#58-同策略与异策略).

## Actor, Critic e vantaggio \{#advantage}

L'**Actor** sceglie le azioni; il **Critic** stima il valore. Puoi pensarli come una parte che decide e una che assegna punteggi, ricordando però che i punteggi del Critic sono stime apprese, non risposte esatte.

La **funzione di vantaggio (advantage)** indica quanto un'azione sia migliore della media della politica attuale in quello stato: $A_\pi(s,a)=Q_\pi(s,a)-V_\pi(s)$. Un valore dell'azione di $6$ e dello stato di $4$ danno vantaggio $2$; un valore negativo indica un risultato sotto il riferimento. In PPO, $\hat{A}_t$ è una stima del vantaggio, non una ricompensa immediata.

Continua a leggere: [Actor-Critic](/docs/foundations/rl-for-robotics/actor-critic), [PPO](/docs/foundations/rl-for-robotics/ppo).

## Rapporti di probabilità e clipping in PPO \{#ppo-clip}

Il **rapporto di probabilità** $r_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\text{old}}(a_t\mid s_t)$ confronta politica nuova e precedente sulla stessa azione. Se la probabilità di un'azione discreta passa da $0.2$ a $0.3$, il rapporto è $1.5$; per azioni continue si confrontano densità.

Il **clipping** limita gli incentivi nell'obiettivo surrogato. Con vantaggio $2$, $\epsilon=0.2$ e rapporto $1.5$, l'obiettivo PPO-Clip per questo campione è $\min(1.5\times2,1.2\times2)=2.4$. Aumentare ancora il rapporto non aumenta il contributo del campione all'obiettivo.

Con vantaggio negativo, si limita il beneficio derivante dalla riduzione eccessiva della probabilità dell'azione. Non ignorare il $\min$ e non interpretare il clipping come un blocco di tutte le probabilità in un intervallo fisso. Il clipping non garantisce che politica nuova e precedente rimangano vicine.

Continua a leggere: [obiettivo Clip di PPO](/docs/foundations/rl-for-robotics/ppo#1021-clip-约束).

## Riferimenti per approfondire \{#references}

- [Sutton & Barto: Reinforcement Learning, An Introduction](http://incompleteideas.net/book/the-book-2nd.html): i capitoli 3, 4 e 6 trattano MDP, equazioni di Bellman e apprendimento TD.
- [OpenAI Spinning Up: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html): definizioni formali di stati, politiche, ritorni, valori e vantaggi.
- [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html): rapporti di probabilità, obiettivo con clipping e relativi limiti.
