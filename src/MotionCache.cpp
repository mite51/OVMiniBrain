#include "MotionCache.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace ovpp {

namespace {

void read_exact(FILE* f, void* dst, size_t bytes, const char* what) {
    if (std::fread(dst, 1, bytes, f) != bytes)
        throw std::runtime_error(std::string("motion cache truncated reading ") + what);
}

}  // namespace

MotionCache MotionCache::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open motion cache: " + path);

    char magic[4];
    read_exact(f, magic, 4, "magic");
    if (magic[0] != 'P' || magic[1] != 'M' || magic[2] != 'C' || magic[3] != '1') {
        std::fclose(f);
        throw std::runtime_error("Bad motion cache magic (expected PMC1): " + path);
    }

    MotionCache mc;
    int32_t dims[3];
    read_exact(f, dims, sizeof(dims), "dims");
    mc.K = dims[0];
    mc.B = dims[1];
    mc.D = dims[2];
    float meta[3];
    read_exact(f, meta, sizeof(meta), "meta");
    mc.fps = meta[0];
    mc.control_dt = meta[1];
    mc.length = meta[2];

    auto load_arr = [&](std::vector<float>& v, size_t count, const char* what) {
        v.resize(count);
        if (count) read_exact(f, v.data(), count * sizeof(float), what);
    };

    const size_t K = mc.K, B = mc.B, D = mc.D;
    load_arr(mc.body_pos, K * B * 3, "body_pos");
    load_arr(mc.body_rot, K * B * 4, "body_rot");
    load_arr(mc.body_vel, K * B * 3, "body_vel");
    load_arr(mc.body_ang_vel, K * B * 3, "body_ang_vel");
    load_arr(mc.dof_pos, K * D, "dof_pos");
    load_arr(mc.dof_vel, K * D, "dof_vel");

    std::fclose(f);
    return mc;
}

}  // namespace ovpp
