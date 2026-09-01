"""MicroDuck tasks and compatibility hooks for mjlab."""

import os

import warp as wp

# mjlab 1.3.0 reads ``wp.context.runtime.driver_version`` when deciding whether
# graph capture is available.  The ROCm Warp 1.13 fork moved ``context`` under
# ``warp._src`` without keeping the public alias that mjlab 1.3.0 expects.
# Preserve the 1.12 API locally; the guard makes this a no-op for the pinned
# CUDA build and for any future Warp release that restores the alias.
if not hasattr(wp, "context"):
    from warp._src import context as _warp_context

    wp.context = _warp_context

# The ROCm MuJoCo Warp fork has an experimental hipGraph path, while mjlab also
# captures ``mjwarp.step`` from the outside.  Keep both layers eager by default;
# respect an explicit user override so updated forks can still be tested.
if "+rocm" in wp.__version__:
    # The experimental hipGraph path currently produces non-finite MicroDuck
    # state on RDNA4/gfx1201.  Eager HIP stepping is stable; users testing an
    # updated ROCm fork can still opt in explicitly with value ``1``.
    os.environ.setdefault("WP_HIP_GRAPH_ENABLE", "0")

    # ROCm/mujoco_warp@9229bb9 caches the first broadphase candidate list on
    # HIP.  That list is not static: objects that begin above the ground are
    # absent from it forever and can fall through the plane.  Force a fresh
    # broadphase on every step until the fork fixes its cache invalidation.
    import mujoco_warp as mjw
    from mujoco_warp._src import collision_driver

    if not getattr(collision_driver.collision, "_microduck_rocm_compat", False):
        _collision = collision_driver.collision

        def _collision_with_dynamic_broadphase(model, data):
            if getattr(data.qpos.device, "is_hip", False) and hasattr(
                data, "_bvh_cached_ctx"
            ):
                delattr(data, "_bvh_cached_ctx")
            return _collision(model, data)

        _collision_with_dynamic_broadphase._microduck_rocm_compat = True
        collision_driver.collision = _collision_with_dynamic_broadphase
        if mjw.collision is _collision:
            mjw.collision = _collision_with_dynamic_broadphase

    # mjlab 1.3.0 wraps the entire physics step in another Warp graph.  The
    # ROCm fork performs host-side convergence checks and secondary-stream
    # synchronization, neither of which can be nested inside mjlab's capture.
    # Keep mjlab eager on HIP; the fork may still own graphing when a user opts
    # into its experimental path above.
    from mjlab.sim.sim import Simulation

    if not getattr(
        Simulation._should_use_cuda_graph, "_microduck_rocm_compat", False
    ):
        _should_use_cuda_graph = Simulation._should_use_cuda_graph

        def _should_use_accelerator_graph(self: Simulation) -> bool:
            if self.wp_device.is_hip:
                return False
            return _should_use_cuda_graph(self)

        _should_use_accelerator_graph._microduck_rocm_compat = True
        Simulation._should_use_cuda_graph = _should_use_accelerator_graph
