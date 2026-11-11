# Codice di supporto ROS2

[中文](README.md) | [English](README.en.md)

Questi file raccolgono gli esempi delle [lezioni 2–11](../../../docs/foundations/robotics-and-ros2/0.intro.md), con la licenza CC BY-NC-SA 4.0 del repository. La vecchia directory `ros_learning` non è più un prerequisito: basta clonare questo repository.

## Struttura e ambiente

```text
ros_ws/
├── scripts/                 # Lezioni 2–3: esempi Python autonomi
├── src/                     # I tuoi pacchetti di esercitazione; inizialmente vuota
└── reference/
    ├── COLCON_IGNORE        # Evita la scoperta di pacchetti duplicati
    ├── moveit/              # Lezione 10: demo_sim.launch.py
    └── src/
        ├── my_robot_basics/       # Lezioni 4–8: nodi, comunicazione, parametri, tf2
        ├── my_robot_interfaces/   # Lezione 6: SolveIK.srv e MoveTo.action
        ├── my_robot_description/  # Lezioni 9–10: URDF, visualizzazione, hardware simulato
        ├── my_robot_planner/      # Lezioni 10–11: pianificazione, obiettivi, ostacoli
        └── my_robot_bringup/      # Lezione 11: avvio del sistema
```

Gli esperimenti ROS richiedono **Ubuntu 24.04, ROS2 Jazzy, Python 3.12 di sistema e Bash**. Installa ROS, colcon e rosdep seguendo la [lezione 1](../../../docs/foundations/robotics-and-ros2/1.course_introduction.md) (in cinese). Le lezioni 2–3 richiedono solo Python e NumPy e possono essere eseguite su macOS, Windows o Linux. Le dipendenze npm del sito non installano quelle Python o ROS.

Dalla radice del repository:

```bash
export ROS_WS="$(pwd)/codes/foundations/robotics-and-ros2/ros_ws"
echo "$ROS_WS"
```

Imposta questa variabile in ogni nuovo terminale, oppure aggiungi `export ROS_WS="..."` a `~/.bashrc` usando il percorso assoluto visualizzato. Conserva `reference/COLCON_IGNORE`. Git ignora gli esercizi e le directory `build/`, `install/` e `log/`.

## Eseguire gli script di cinematica

Crea un ambiente separato per gli script matematici e i controlli; non usarlo per compilare pacchetti ROS:

```bash
cd codes/foundations/robotics-and-ros2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python "$ROS_WS/scripts/lesson_02_transforms.py"
python "$ROS_WS/scripts/lesson_03_two_link_kinematics.py"
python -m unittest discover -s tests -v
deactivate
```

## Lezioni 4–9: esercizi e pacchetti di riferimento

Crea i pacchetti in `$ROS_WS/src` con `ros2 pkg create` seguendo le lezioni, poi confronta i singoli file con `reference/src`. La versione di riferimento integra le modifiche successive: parametri e frequenza di pubblicazione dinamica della lezione 7, parametri di avvio del server Action e tag `ros2_control` della lezione 10 nell'URDF. Non inserire di nuovo questi tag dopo aver copiato l'URDF. `SolveIK` usa le lunghezze dei bracci indicate in ogni richiesta, non valori predefiniti YAML. L'esempio Action mostra obiettivi, feedback e risultati; la cancellazione non è implementata.

Per provare prima i pacchetti di riferimento, copiali in uno spazio di lavoro **senza pacchetti di esercitazione con gli stessi nomi**. Se hai già svolto degli esercizi, confronta o copia solo i singoli file necessari per conservare il tuo lavoro.

```bash
source /opt/ros/jazzy/setup.bash
cd "$ROS_WS"
cp -rn reference/src/my_robot_interfaces reference/src/my_robot_basics reference/src/my_robot_description src/
rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y
colcon build --symlink-install --packages-up-to my_robot_basics my_robot_description
source install/setup.bash
ros2 run my_robot_basics hello_node
```

Interrompi `hello_node` con Ctrl+C. Altri comandi, da eseguire separatamente secondo necessità:

```bash
ros2 launch my_robot_basics demo.launch.py publish_rate:=2.0
ros2 launch my_robot_description display.launch.py
```

In un nuovo terminale carica `/opt/ros/jazzy/setup.bash` e `"$ROS_WS/install/setup.bash"` con `source`. Segui le lezioni 8–9 passando dagli esperimenti TF manuali a quelli URDF; evita di pubblicare le stesse trasformazioni sia dal tuo broadcaster sia da `robot_state_publisher`.

## Lezioni 10–11: generare la configurazione MoveIt

`my_robot_moveit_config` **non è incluso**. Genera il pacchetto in `$ROS_WS/src/my_robot_moveit_config` con MoveIt Setup Assistant, come descritto nella [lezione 10](../../../docs/foundations/robotics-and-ros2/10.moveit2_basics.md), e completa la configurazione di SRDF, pianificatore, limiti dei giunti e controllori. Copia poi il file di avvio e i due pacchetti restanti, solo se non esistono già esercizi con gli stessi nomi:

```bash
cd "$ROS_WS"
cp -n reference/moveit/demo_sim.launch.py src/my_robot_moveit_config/launch/
cp -rn reference/src/my_robot_planner reference/src/my_robot_bringup src/
rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch my_robot_bringup full_system.launch.py
```

Da un altro terminale con l'ambiente caricato:

```bash
ros2 topic pub --once /target_pose geometry_msgs/msg/Point "{x: 1.2, y: 0.8, z: 0.0}"
```

## Ambito dei controlli

`tests/` verifica risultati numerici, entrambe le soluzioni IK e i limiti dello spazio di lavoro, sintassi Python, punti di ingresso e risorse dei pacchetti e struttura URDF. Questi controlli non caricano ROS e non verificano l'esecuzione di nodi, RViz2 o MoveIt. Gli esperimenti completi vanno eseguiti su Ubuntu / Jazzy seguendo le lezioni.
