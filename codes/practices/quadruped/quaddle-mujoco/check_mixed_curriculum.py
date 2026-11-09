"""Check reset-only map sampling, deterministic resets and bounded model cache."""
import numpy as np
from mixed_curriculum import MixedCurriculumEnv

env=MixedCurriculumEnv(level=2,seconds=.01)
a,ia=env.reset(seed=123)
b,ib=env.reset(seed=123)
np.testing.assert_array_equal(a,b)
assert ia==ib
observed=set()
for seed in range(12):
    obs,info=env.reset(seed=seed)
    observed.add((info['level'],info['terrain_seed']))
    assert len(env.cache)<=2 and env.data.time>=1.999
    key=(env.active_level,env.active_terrain_seed)
    _,reward,terminated,truncated,info=env.step(np.zeros(4))
    assert key==(env.active_level,env.active_terrain_seed)
    assert np.isfinite(reward) and truncated and not terminated
    assert not np.any(env.data.xfrc_applied) and not np.any(env.data.qfrc_applied)
assert {x[0] for x in observed}=={1,2}
assert {x[1] for x in observed}=={7,11,13,17}
env.close()
print('Mixed curriculum deterministic reset / sampling / cache / unchanged rollout terrain PASS')
