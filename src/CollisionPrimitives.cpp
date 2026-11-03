#include "CollisionPrimitives.h"

#include <algorithm>
#include <cmath>

namespace ovpp {

void CollisionPrimitives::resize(int m) {
    M = m;
    pos.assign((size_t)m * 3, 0.0f);
    rot.assign((size_t)m * 4, 0.0f);
    lin_vel.assign((size_t)m * 3, 0.0f);
    radius.assign(m, 0.0f);
    extent_z.assign(m, 0.0f);
    damage.assign(m, 0.0f);
    mass.assign(m, 0.0f);
    shape.assign((size_t)m * 2, 0.0f);
    valid.assign(m, 0.0f);
    // Identity quaternion (xyzw) in every slot; only projectiles overwrite it.
    for (int i = 0; i < m; ++i) rot[(size_t)i * 4 + 3] = 1.0f;
}

namespace {

// A single raw candidate before capacity compaction.
struct Candidate {
    std::array<float, 3> pos{{0, 0, 0}};
    std::array<float, 4> rot{{0, 0, 0, 1}};
    std::array<float, 3> lin_vel{{0, 0, 0}};
    float radius = 0.0f;
    float extent_z = 0.0f;
    float damage = 0.0f;
    float mass = 0.0f;
    std::array<float, 2> shape{{0, 0}};
    float valid = 0.0f;
};

// Port of compute_collision_priority: score each candidate against every body of
// the observing character by proximity + closing speed + mass. Invalid / out of
// range candidates get a large negative score so they sort last.
float priorityScore(const Candidate& c, int candidate_index,
                    const ArticulationState& self_state,
                    const CollisionSelectionParams& p) {
    const int nb = static_cast<int>(self_state.body_pos.size());
    float nearest = 1.0e30f;
    float max_closing = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const auto& bp = self_state.body_pos[b];
        const auto& bv = self_state.body_lin_vel[b];
        float dx = bp[0] - c.pos[0];
        float dy = bp[1] - c.pos[1];
        float dz = bp[2] - c.pos[2];
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < 1.0e-6f) dist = 1.0e-6f;
        if (dist < nearest) nearest = dist;
        // Closing speed = component of (primitive_vel - body_vel) toward the body.
        float rvx = c.lin_vel[0] - bv[0];
        float rvy = c.lin_vel[1] - bv[1];
        float rvz = c.lin_vel[2] - bv[2];
        float toward = (rvx * dx + rvy * dy + rvz * dz) / dist;
        if (toward > max_closing) max_closing = toward;
    }

    float proximity = p.distance_scale / (nearest + p.distance_scale);
    float closing = max_closing / (max_closing + p.speed_scale);
    float m = c.mass < 0.0f ? 0.0f : c.mass;
    float mass_priority = m / (m + p.mass_scale);
    float score = p.distance_weight * proximity + p.closing_speed_weight * closing +
                  p.mass_weight * mass_priority;

    bool eligible = (c.valid > 0.5f) && (nearest <= p.selection_range);
    if (!eligible) score = -1.0e9f;
    // Stable tiebreak: nearer, then earlier candidate.
    return score - nearest * 1.0e-6f - static_cast<float>(candidate_index) * 1.0e-8f;
}

void writeSlot(CollisionPrimitives& out, int slot, const Candidate& c) {
    out.pos[(size_t)slot * 3 + 0] = c.pos[0];
    out.pos[(size_t)slot * 3 + 1] = c.pos[1];
    out.pos[(size_t)slot * 3 + 2] = c.pos[2];
    out.rot[(size_t)slot * 4 + 0] = c.rot[0];
    out.rot[(size_t)slot * 4 + 1] = c.rot[1];
    out.rot[(size_t)slot * 4 + 2] = c.rot[2];
    out.rot[(size_t)slot * 4 + 3] = c.rot[3];
    out.lin_vel[(size_t)slot * 3 + 0] = c.lin_vel[0];
    out.lin_vel[(size_t)slot * 3 + 1] = c.lin_vel[1];
    out.lin_vel[(size_t)slot * 3 + 2] = c.lin_vel[2];
    out.radius[slot] = c.radius;
    out.extent_z[slot] = c.extent_z;
    out.damage[slot] = c.damage;
    out.mass[slot] = c.mass;
    out.shape[(size_t)slot * 2 + 0] = c.shape[0];
    out.shape[(size_t)slot * 2 + 1] = c.shape[1];
    out.valid[slot] = c.valid;
}

}  // namespace

void buildCollisionPrimitives(CollisionPrimitives& out, int M,
                              const ArticulationState& self_state, float ground_z,
                              const ProjectileSnapshot* proj,
                              const std::vector<const ArticulationState*>& opponents,
                              const CollisionSceneInfo& scene) {
    out.resize(M);

    std::vector<Candidate> raw;
    const int P = (proj != nullptr) ? proj->P : 0;
    const int keyCount = static_cast<int>(scene.opponent_key_body_ids.size());
    const int oppCount = static_cast<int>(opponents.size()) * keyCount;
    raw.reserve(1 + P + oppCount);

    // Slot 0: ground box beneath the root.
    {
        Candidate g;
        g.pos = {self_state.root_pos[0], self_state.root_pos[1], ground_z};
        g.shape = {1.0f, 0.0f};  // is_box
        g.damage = scene.ground_damage;
        g.mass = scene.static_mass;
        g.valid = 1.0f;
        raw.push_back(g);
    }

    // Slots 1..P: active projectiles (velocity supplied by the pool via finite
    // difference; hidden slots carry valid == 0).
    for (int i = 0; i < P; ++i) {
        Candidate c;
        c.pos = proj->pos[i];
        c.rot = proj->rot[i];
        c.lin_vel = proj->lin_vel[i];
        c.radius = proj->radius[i];
        c.extent_z = proj->extent_z[i];
        c.shape = proj->shape[i];
        c.damage = proj->damage[i];
        c.mass = proj->mass[i];
        c.valid = proj->active[i];
        raw.push_back(c);
    }

    // Opponents' key bodies as spheres.
    for (const ArticulationState* opp : opponents) {
        for (int kb : scene.opponent_key_body_ids) {
            Candidate c;
            c.pos = opp->body_pos[kb];
            c.lin_vel = opp->body_lin_vel[kb];
            c.radius = scene.opponent_body_radius;
            c.shape = {0.0f, 1.0f};  // sphere one-hot
            c.damage = 0.0f;
            c.mass = (kb < (int)scene.body_mass.size()) ? scene.body_mass[kb] : 1.0f;
            c.valid = 1.0f;
            raw.push_back(c);
        }
    }

    const int rawCap = static_cast<int>(raw.size());
    if (rawCap <= M) {
        for (int i = 0; i < rawCap; ++i) writeSlot(out, i, raw[i]);
        return;  // remaining slots stay zeroed with valid == 0
    }

    // Compact to the M highest-priority candidates (mirrors the training-time
    // capacity cut before the ONNX top-K).
    std::vector<int> idx(rawCap);
    for (int i = 0; i < rawCap; ++i) idx[i] = i;
    std::vector<float> score(rawCap);
    for (int i = 0; i < rawCap; ++i)
        score[i] = priorityScore(raw[i], i, self_state, scene.selection);
    std::partial_sort(idx.begin(), idx.begin() + M, idx.end(),
                      [&](int a, int b) { return score[a] > score[b]; });
    for (int i = 0; i < M; ++i) writeSlot(out, i, raw[idx[i]]);
}

}  // namespace ovpp
