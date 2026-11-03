#include "RobotChain.h"

#include <cmath>

namespace ovpp {

namespace {
const std::array<float, 3> kComOffsets[24] = {
    {-0.005500f, 0.000000f, -0.012100f},  // Pelvis
    {-0.002250f, 0.017150f, -0.187600f},  // L_Hip
    {-0.021850f, -0.006800f, -0.199000f},  // L_Knee
    {0.024200f, 0.023300f, -0.023900f},    // L_Ankle
    {0.024800f, -0.003000f, 0.005500f},    // L_Toe
    {-0.004450f, -0.019150f, -0.191300f},  // R_Hip
    {-0.021150f, 0.007900f, -0.199200f},   // R_Knee
    {0.025600f, -0.021200f, -0.017400f},   // R_Ankle
    {0.022700f, 0.004200f, 0.004500f},     // R_Toe
    {0.000550f, 0.002750f, 0.067550f},     // Torso
    {0.012700f, 0.000750f, 0.026450f},     // Spine
    {-0.019250f, -0.000950f, 0.075750f},   // Chest
    {0.025700f, 0.002550f, 0.032500f},     // Neck
    {-0.011600f, -0.004200f, 0.087600f},   // Head
    {-0.004450f, 0.045500f, 0.015250f},    // L_Thorax
    {-0.013750f, 0.129800f, -0.006400f},   // L_Shoulder
    {-0.000550f, 0.124600f, 0.004500f},    // L_Elbow
    {-0.007500f, 0.042000f, -0.004050f},   // L_Wrist
    {-0.005800f, 0.049300f, 0.001000f},    // L_Hand
    {-0.004550f, -0.048000f, 0.016250f},   // R_Thorax
    {-0.010700f, -0.126850f, -0.006700f},  // R_Shoulder
    {-0.002750f, -0.127650f, 0.003900f},   // R_Elbow
    {-0.005200f, -0.042300f, -0.003050f},  // R_Wrist
    {-0.007900f, -0.046200f, -0.000900f},  // R_Hand
};
}  // namespace

const std::array<float, 3>* smplComOffsets() { return kComOffsets; }
int smplComOffsetCount() { return 24; }

std::array<float, 3> quatFdOmega(const std::array<float, 4>& qp,
                                 const std::array<float, 4>& qc_in, float dt) {
    std::array<float, 4> qc = qc_in;
    float dot = qp[0] * qc[0] + qp[1] * qc[1] + qp[2] * qc[2] + qp[3] * qc[3];
    if (dot < 0) { for (auto& v : qc) v = -v; }
    // qd_xyz = vc*wp - vp*wc - cross(vp, vc), XYZW layout.
    float vpx = qp[0], vpy = qp[1], vpz = qp[2], wp = qp[3];
    float vcx = qc[0], vcy = qc[1], vcz = qc[2], wc = qc[3];
    float cx = vpy * vcz - vpz * vcy;
    float cy = vpz * vcx - vpx * vcz;
    float cz = vpx * vcy - vpy * vcx;
    float qx = vcx * wp - vpx * wc - cx;
    float qy = vcy * wp - vpy * wc - cy;
    float qz = vcz * wp - vpz * wc - cz;
    float k = 2.0f / dt;
    return {k * qx, k * qy, k * qz};
}

std::array<float, 3> applyComCorrection(const std::array<float, 3>& v,
                                        const std::array<float, 3>& w,
                                        const std::array<float, 4>& q,
                                        const std::array<float, 3>& com) {
    float x = q[0], y = q[1], z = q[2], wq = q[3];
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = wq * x, wy = wq * y, wz = wq * z;
    // R . com
    float rx = (1 - 2 * (yy + zz)) * com[0] + 2 * (xy - wz) * com[1] + 2 * (xz + wy) * com[2];
    float ry = 2 * (xy + wz) * com[0] + (1 - 2 * (xx + zz)) * com[1] + 2 * (yz - wx) * com[2];
    float rz = 2 * (xz - wy) * com[0] + 2 * (yz + wx) * com[1] + (1 - 2 * (xx + yy)) * com[2];
    // omega x (R.com)
    float cx = w[1] * rz - w[2] * ry;
    float cy = w[2] * rx - w[0] * rz;
    float cz = w[0] * ry - w[1] * rx;
    return {v[0] + cx, v[1] + cy, v[2] + cz};
}

}  // namespace ovpp
