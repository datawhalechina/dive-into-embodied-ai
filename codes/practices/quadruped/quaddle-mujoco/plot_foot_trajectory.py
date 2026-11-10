"""Plot measured collision-mesh lowest vertices, not an assumed spherical foot."""
import csv,json
from pathlib import Path
from curriculum_rl import curriculum_scene
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('outputs/foot_trajectory_20261110')
def read(name):
 rows=list(csv.DictReader((ROOT/name/'feet.csv').open()))
 return [{k:(v if k in ('leg','geometry_refreshed') else float(v) if v else np.nan) for k,v in x.items()} for x in rows]
def vals(rows,key):return [x[key] for x in rows]
if __name__=='__main__':
 rows=read('stairs_2mm');leg=[x for x in rows if x['leg']=='left_back' and 7.5<=x['time_s']<=8.15]
 peak=json.loads((ROOT/'stairs_2mm/peak.json').read_text());t=vals(leg,'time_s')
 fig,ax=plt.subplots(4,1,figsize=(11,10),sharex=True)
 ax[0].plot(t,vals(leg,'dq'),label='Actual joint velocity');ax[0].axhline(10,c='r',ls='--',label='10 rad/s limit');ax[0].scatter([peak['sample']['time_s']],[peak['speed_rad_s']],c='orange',s=35,label='1 kHz peak: %.3f rad/s'%peak['speed_rad_s']);ax[0].set_ylabel('rad/s');ax[0].legend()
 ax[1].plot(t,vals(leg,'q'),label='Actual angle');ax[1].plot(t,vals(leg,'target'),label='Target angle');ax[1].set_ylabel('rad');ax[1].legend()
 ax[2].plot(t,vals(leg,'normal_force_N'),label='Sum of normal contact forces');ax[2].fill_between(t,0,vals(leg,'riser'),alpha=.2,label='Riser contact count');ax[2].set_ylabel('N / count');ax[2].legend()
 ax[3].plot(t,vals(leg,'torque_Nm'),label='Actuator torque');ax[3].set_ylabel('Nm');ax[3].set_xlabel('Simulation time (s)');ax[3].legend()
 for a in ax:a.axvline(peak['sample']['time_s'],c='orange',ls=':');a.grid(alpha=.2)
 fig.suptitle('2 mm stairs: left rear leg accelerates after contact release\n200 Hz trace; 1 kHz peak detection');fig.tight_layout();fig.savefig(ROOT/'speed_peak.png',dpi=180);fig.savefig(ROOT/'speed_peak.svg');plt.close(fig)
 rows=read('single_4mm');fig,ax=plt.subplots(3,1,figsize=(11,11))
 for side in ['left_front','right_front','right_back','left_back']:
  r=[x for x in rows if x['leg']==side and 3.5<=x['time_s']<=10.]
  ax[0].plot(vals(r,'foot_forward_m'),np.array(vals(r,'foot_bottom_z_m'))*1000,label=side,lw=.8)
  ax[1].plot(vals(r,'time_s'),vals(r,'normal_force_N'),label=side,lw=.8)
 step=curriculum_scene('stairs_up',3,single_step=True).boxes[0]
 ax[0].plot([-.1,step['s0'],step['s0'],.5],[0,0,step['top']*1000,step['top']*1000],'k--',label='Single step profile')
 ax[0].set(xlabel='World forward coordinate (m)',ylabel='Lowest collision vertex z (mm)',ylim=(-2,40));ax[0].legend(ncol=3)
 ax[1].set(xlabel='Time (s)',ylabel='Foot normal force (N)');ax[1].legend(ncol=4)
 r=[x for x in rows if x['leg']=='left_front' and 3.5<=x['time_s']<=10.]
 ax[2].plot(vals(r,'time_s'),vals(r,'pitch_deg'),label='Body pitch');ax[2].plot(vals(r,'time_s'),vals(r,'yaw_deg'),label='Body yaw');ax[2].set(xlabel='Time (s)',ylabel='deg');ax[2].legend()
 for a in ax:a.grid(alpha=.2)
 fig.suptitle('4 mm single step: measured foot envelopes, contact forces and body pose\nLowest vertex can switch; this is not a fixed material point or an IK workspace')
 fig.tight_layout();fig.savefig(ROOT/'single_step_trajectory.png',dpi=180);fig.savefig(ROOT/'single_step_trajectory.svg')
