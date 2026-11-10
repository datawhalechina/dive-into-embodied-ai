"""Separate rough-ground diagnostic with explicitly bounded neighboring height jumps."""
import numpy as np
from scenarios import make_scene

def graded_rough(seed=7,max_jump_m=.002):
    if not 0<max_jump_m<=.006:raise ValueError('max_jump_m must be in (0, .006]')
    scene=make_scene('rough')
    rng=np.random.default_rng(seed);height=rng.uniform(-.003,.003,(12,8))
    # Spatially correlated heights, not independent stair edges at each cell.
    for _ in range(2):
        padded=np.pad(height,1,mode='edge')
        height=sum(padded[i:i+12,j:j+8] for i in range(3) for j in range(3))/9
    height*=.003/max(np.max(np.abs(height)),1e-12)
    # Include transitions to the fixed start/finish platforms in the bound.
    jump=max(np.max(np.abs(np.diff(height,axis=0))),np.max(np.abs(np.diff(height,axis=1))),np.max(np.abs(height[0])),np.max(np.abs(height[-1])))
    height*=min(1.,max_jump_m/jump)
    for box,z in zip([b for b in scene.boxes if b['name'].startswith('tile_')],height.flat):box['top']=float(z)
    scene.seed=seed;scene.label=f'Graded rough / adjacent height jump <= {max_jump_m*1000:g}mm / seed {seed}'
    return scene

def geometry_stats(scene):
    h=np.array([b['top'] for b in scene.boxes if b['name'].startswith('tile_')]).reshape(12,8)
    return dict(min_height_m=float(h.min()),max_height_m=float(h.max()),max_neighbor_jump_m=float(max(np.abs(np.diff(h,axis=0)).max(),np.abs(np.diff(h,axis=1)).max())),max_entry_exit_jump_m=float(max(abs(h[0]).max(),abs(h[-1]).max())))
