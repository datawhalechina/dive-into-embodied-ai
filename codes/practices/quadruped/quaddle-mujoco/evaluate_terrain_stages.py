"""Full length unchanged matrix checks, independent of training termination/reward."""
import argparse,json
from pathlib import Path
from scenario_matrix import run_case


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args()
    import torch
    torch.set_num_threads(1)
    r=run_case(a.scene,'walk',output=a.output,video=a.video,policy_path=a.policy)
    print(json.dumps({k:r[k] for k in ['passed','forward_m','checks','max_heading_deviation_deg','max_driven_joint_speed_rad_s']},indent=2),flush=True)

if __name__=='__main__': main()
