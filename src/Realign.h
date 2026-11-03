// Smooth, velocity-error-gated reference re-anchoring.
//
// Port of compute_smooth_realign_offset (protomotions/envs/base_env/utils.py).
// Maintains a persistent per-character reference XY offset that lerps toward the
// instant-snap target at a rate gated by the "unexpected" root velocity error, so
// deliberate clip motion (getup, locomotion) leaves the anchor put while shoves
// pull it along. The offset is added to mimic_future_pos (XY only); the fight
// checkpoint trained with this on.
#pragma once

#include <array>
#include <cmath>

namespace ovpp {

struct RealignParams {
    float alpha_min = 0.0f;
    float alpha_max = 0.4f;
    float vel_err_low = 0.3f;    // m/s
    float vel_err_high = 1.5f;   // m/s
    float max_xy_speed = 2.0f;   // m/s cap on offset drift
};

// Return the updated reference offset XY for one character.
inline std::array<float, 2> computeSmoothRealignOffset(
    const std::array<float, 2>& prev_offset_xy,
    const std::array<float, 2>& current_root_xy,
    const std::array<float, 2>& current_root_xy_vel,
    const std::array<float, 2>& ref_root_xy,
    const std::array<float, 2>& ref_root_xy_vel, const RealignParams& p, float dt) {
    const float eps = 1e-6f;
    const std::array<float, 2> target = {current_root_xy[0] - ref_root_xy[0],
                                         current_root_xy[1] - ref_root_xy[1]};

    const float dvx = current_root_xy_vel[0] - ref_root_xy_vel[0];
    const float dvy = current_root_xy_vel[1] - ref_root_xy_vel[1];
    const float vel_err = std::sqrt(dvx * dvx + dvy * dvy);

    const float denom = std::max(p.vel_err_high - p.vel_err_low, eps);
    float t = (vel_err - p.vel_err_low) / denom;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    t = t * t * (3.0f - 2.0f * t);  // smoothstep
    const float alpha = p.alpha_min + (p.alpha_max - p.alpha_min) * t;

    std::array<float, 2> new_xy = {
        prev_offset_xy[0] + alpha * (target[0] - prev_offset_xy[0]),
        prev_offset_xy[1] + alpha * (target[1] - prev_offset_xy[1])};

    // Cap the per-step drift.
    float dx = new_xy[0] - prev_offset_xy[0];
    float dy = new_xy[1] - prev_offset_xy[1];
    float dn = std::sqrt(dx * dx + dy * dy);
    float max_delta = p.max_xy_speed * dt;
    if (dn > max_delta && dn > eps) {
        float scale = max_delta / dn;
        dx *= scale;
        dy *= scale;
    }
    return {prev_offset_xy[0] + dx, prev_offset_xy[1] + dy};
}

}  // namespace ovpp
