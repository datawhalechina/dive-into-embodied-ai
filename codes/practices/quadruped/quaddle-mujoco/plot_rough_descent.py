"""Make terrain comparison plots; separate geometry change from controller change."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scenarios import make_scene
from graded_rough import graded_rough,geometry_stats
ROOT=Path('outputs/rough_descent_20261110')
if __name__=='__main__':
 fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
 for ax,scene,title in zip(axes,[make_scene('rough'),graded_rough(7)],['Original independent tile heights','Separate graded rough diagnostic']):
  h=np.array([b['top'] for b in scene.boxes if b['name'].startswith('tile_')]).reshape(12,8)*1000
  im=ax.imshow(h,origin='lower',extent=(-.24,.24,.18,.9),vmin=-3,vmax=3,cmap='coolwarm',aspect='equal')
  stats=geometry_stats(scene);ax.set(title=title+'\nMax internal height jump: %.2f mm'%(stats['max_neighbor_jump_m']*1000),xlabel='Lateral x (m)',ylabel='Forward (m)')
 fig.colorbar(im,ax=axes,label='Tile height (mm)');fig.savefig(ROOT/'terrain_comparison.png',dpi=180);fig.savefig(ROOT/'terrain_comparison.svg');plt.close(fig)
 fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,constrained_layout=True)
 for name,label in [('rough_prior_ppo','Existing PPO on original rough'),('rough_transfer','2mm stair controller on original rough')]:
  r=list(csv.DictReader((ROOT/name/'feet.csv').open()));r=[x for x in r if x['leg']=='left_front']
  axes[0].plot([float(x['time_s']) for x in r],[float(x['body_forward_m']) for x in r],label=label)
  axes[1].plot([float(x['time_s']) for x in r],[float(x['pitch_deg']) for x in r],label=label)
 axes[0].set_ylabel('World forward position (m)');axes[1].set(ylabel='Body pitch (deg)',xlabel='Time (s)')
 for ax in axes:ax.grid(alpha=.2);ax.legend()
 fig.suptitle('Original rough ground: progress and body attitude (no terrain change)')
 fig.savefig(ROOT/'rough_diagnosis.png',dpi=180);fig.savefig(ROOT/'rough_diagnosis.svg')
