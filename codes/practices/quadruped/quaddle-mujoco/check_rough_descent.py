"""Check diagnostic parity and bounded-jump terrain geometry without weakening physics."""
import json
from pathlib import Path
import numpy as np
from graded_rough import graded_rough,geometry_stats
from scenarios import make_scene
for seed in [7,19,23]:
 a=graded_rough(seed);b=graded_rough(seed);assert a.boxes==b.boxes
 s=geometry_stats(a)
 assert s['max_neighbor_jump_m']<=.002+1e-12 and s['max_entry_exit_jump_m']<=.002+1e-12
 assert s['min_height_m']>=-.003-1e-12 and s['max_height_m']<=.003+1e-12
 assert a.goal_m==make_scene('rough').goal_m and a.plane_z==make_scene('rough').plane_z
assert geometry_stats(make_scene('rough'))['max_neighbor_jump_m']>.005
print('deterministic graded geometry / internal and entry-exit jump bounds / original retained PASS')
old=json.loads(Path('outputs/practice_20261110/videos/prior/walk/rough/metrics.json').read_text())
new=json.loads(Path('outputs/rough_descent_20261110/rough_prior_ppo/walk/rough/metrics.json').read_text())
for key in ['forward_m','checks','max_driven_joint_speed_rad_s','max_heading_deviation_deg','body_contact_samples']:
 assert old[key]==new[key],key
print('generic observer: prior PPO physical metrics unchanged PASS')
