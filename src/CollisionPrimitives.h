// World-frame collision-primitive candidate buffer for the fighting-mimic obs.
//
// Port of BaseEnv._build_collision_primitives / _write_opponent_primitives
// (protomotions/envs/base_env/env.py) and compute_collision_priority
// (protomotions/envs/obs/collision_primitives.py). The ONNX graph does the
// egocentric transform + top-K selection internally, so the client only fills
// these M world-frame slots per observing character:
//
//   slot 0        : ground box beneath the root (rel_pos.z encodes root height)
//   slots 1..P    : active thrown projectiles (finite-difference velocity)
//   next slots    : opponents' key bodies as spheres (multi-character tier)
//
// When the raw candidate count exceeds M the same contact-priority score used in
// training compacts the set to the M highest-priority slots.
#pragma once

#include "SmplArticulation.h"

#include <array>
#include <vector>

namespace ovpp {

// Tunables mirrored from EnvConfig (collision_* fields). Defaults match the
// deployed smpl_fight_multichar config.
struct CollisionSelectionParams {
    float selection_range = 8.0f;
    float distance_weight = 1.0f;
    float closing_speed_weight = 1.0f;
    float mass_weight = 0.25f;
    float distance_scale = 2.0f;
    float speed_scale = 10.0f;
    float mass_scale = 10.0f;
};

// One control step's projectile state (all P pool slots), shared by every
// character in the scene. Geometry arrays are static per pool index; pose /
// velocity / active update each step.
struct ProjectileSnapshot {
    int P = 0;
    std::vector<std::array<float, 3>> pos;      // [P] world
    std::vector<std::array<float, 4>> rot;      // [P] xyzw
    std::vector<std::array<float, 3>> lin_vel;  // [P] finite-difference world vel
    std::vector<float> active;                  // [P] 1.0 active / 0.0 hidden
    std::vector<float> radius;                  // [P]
    std::vector<float> extent_z;                // [P]
    std::vector<std::array<float, 2>> shape;    // [P] [is_box, is_sphere]
    std::vector<float> damage;                  // [P]
    std::vector<float> mass;                    // [P]
};

// The M-slot world-frame buffer fed (as 9 separate ONNX inputs) for one
// observing character.
struct CollisionPrimitives {
    int M = 0;
    std::vector<float> pos;       // [M,3]
    std::vector<float> rot;       // [M,4] xyzw
    std::vector<float> lin_vel;   // [M,3]
    std::vector<float> radius;    // [M]
    std::vector<float> extent_z;  // [M]
    std::vector<float> damage;    // [M]
    std::vector<float> mass;      // [M]
    std::vector<float> shape;     // [M,2]
    std::vector<float> valid;     // [M]

    void resize(int m);
};

// Static per-scene inputs to the builder (independent of the observing char).
struct CollisionSceneInfo {
    float ground_z = 0.0f;                    // terrain height under the root
    float ground_damage = 0.0f;               // EnvConfig.ground_primitive_damage
    float static_mass = 100.0f;               // EnvConfig.static_collider_effective_mass
    float opponent_body_radius = 0.12f;       // EnvConfig.opponent_body_radius
    std::vector<int> opponent_key_body_ids;   // ascending body indices (10 for SMPL)
    std::vector<float> body_mass;             // [num_bodies] per-link mass (opponent slots)
    CollisionSelectionParams selection;
};

// Fill ``out`` (M slots) for the observing character.
//   self_state  : observing character's current body state (root == index 0)
//   proj        : projectile snapshot (nullptr / P==0 when no projectiles)
//   opponents   : other characters' current body state in the same scene
void buildCollisionPrimitives(CollisionPrimitives& out, int M,
                              const ArticulationState& self_state, float ground_z,
                              const ProjectileSnapshot* proj,
                              const std::vector<const ArticulationState*>& opponents,
                              const CollisionSceneInfo& scene);

}  // namespace ovpp
