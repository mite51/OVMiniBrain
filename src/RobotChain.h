// SMPL COM offsets + finite-difference / COM-correction velocity helpers.
//
// Port of scripts/robot_chain.py. With raw PhysX 5.6.1 the articulation cache
// reports link velocities at the COM in world space already (IsaacLab's
// convention), so the default read path uses those directly. This module is
// kept as a fallback for diagnosing body-velocity convention mismatches.
#pragma once

#include <array>
#include <vector>

namespace ovpp {

// SMPL per-body COM offset in the body-local link frame (body_names order).
const std::array<float, 3>* smplComOffsets();  // returns pointer to [24]
int smplComOffsetCount();

// omega = 2 * imag(q_curr (x) conj(q_prev)) / dt, shortest-path corrected.
std::array<float, 3> quatFdOmega(const std::array<float, 4>& q_prev_xyzw,
                                 const std::array<float, 4>& q_curr_xyzw, float dt);

// v_at_COM = v_at_link + omega x (R(q) . com_offset_local).
std::array<float, 3> applyComCorrection(const std::array<float, 3>& v_at_link,
                                        const std::array<float, 3>& omega,
                                        const std::array<float, 4>& q_xyzw,
                                        const std::array<float, 3>& com_offset_local);

}  // namespace ovpp
