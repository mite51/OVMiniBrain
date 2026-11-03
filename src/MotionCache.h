// Reader for the precomputed motion cache (.pmc) produced by
// tools/dump_motion_cache.py.
//
// The cache is sampled at exact control_dt ticks straight from MotionLib, so
// indexing it by control-dt tick is byte-identical to a live MotionLib query
// (no C++ interpolation). All data is sim coords, xyzw quats, no re-rooting.
#pragma once

#include <string>
#include <vector>

namespace ovpp {

class MotionCache {
public:
    int K = 0;  // number of ticks
    int B = 0;  // bodies
    int D = 0;  // dofs
    float fps = 0.0f;
    float control_dt = 0.0f;
    float length = 0.0f;

    // Flat row-major arrays.
    std::vector<float> body_pos;      // [K,B,3]
    std::vector<float> body_rot;      // [K,B,4] xyzw
    std::vector<float> body_vel;      // [K,B,3]
    std::vector<float> body_ang_vel;  // [K,B,3]
    std::vector<float> dof_pos;       // [K,D]
    std::vector<float> dof_vel;       // [K,D]

    static MotionCache load(const std::string& path);

    int clampTick(int tick) const {
        if (tick < 0) return 0;
        if (tick >= K) return K - 1;
        return tick;
    }

    const float* bodyPos(int tick) const { return &body_pos[(size_t)clampTick(tick) * B * 3]; }
    const float* bodyRot(int tick) const { return &body_rot[(size_t)clampTick(tick) * B * 4]; }
    const float* bodyVel(int tick) const { return &body_vel[(size_t)clampTick(tick) * B * 3]; }
    const float* bodyAngVel(int tick) const { return &body_ang_vel[(size_t)clampTick(tick) * B * 3]; }
    const float* dofPos(int tick) const { return &dof_pos[(size_t)clampTick(tick) * D]; }
    const float* dofVel(int tick) const { return &dof_vel[(size_t)clampTick(tick) * D]; }
};

}  // namespace ovpp
