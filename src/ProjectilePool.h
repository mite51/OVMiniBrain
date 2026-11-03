// Thrown-projectile pool (PhysX rigid dynamics) for the fighting-mimic tier.
//
// Port of the projectile system in
// protomotions/simulator/base_simulator/simulator.py (_init_projectiles,
// _throw_projectile, _update_projectiles, get_active_projectile_states) plus the
// per-episode damage randomization in BaseEnv._randomize_projectile_damage. A
// fixed pool of P rigid bodies (box/sphere/capsule round-robin) lives in the
// shared PhysxWorld; each is disabled while hidden and re-enabled when thrown.
//
// Velocity fed to the collision-primitive obs is a finite difference of position
// across control steps (backend-agnostic, matching training), not the PhysX
// solver velocity.
#pragma once

#include "CollisionPrimitives.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ovpp {

class PhysxWorld;

struct ProjectileParams {
    int num_projectiles = 3;
    std::array<float, 2> cube_half_size_range = {0.05f, 0.15f};
    float density = 300.0f;
    std::array<float, 2> speed_range = {12.0f, 22.0f};
    std::array<float, 2> spawn_distance_range = {4.0f, 5.0f};
    std::array<float, 2> spawn_height_range = {-0.65f, 1.1f};
    float direction_noise_std = 0.1f;
    float hide_delay = 2.0f;
    float hide_z = -2.0f;
    std::vector<std::string> shapes = {"box", "sphere", "capsule"};
    float auto_throw_prob = 0.01f;
    // Per-episode damage randomization (BaseEnv._randomize_projectile_damage).
    float baseline_damage = 0.1f;
    std::array<float, 2> damage_range = {0.3f, 1.0f};
    std::array<float, 2> dangerous_fraction_range = {0.0f, 1.0f};
};

class ProjectilePool {
public:
    ProjectilePool(PhysxWorld& world, const ProjectileParams& params, uint64_t seed);
    ~ProjectilePool();

    ProjectilePool(const ProjectilePool&) = delete;
    ProjectilePool& operator=(const ProjectilePool&) = delete;

    // Hide all projectiles, reset timers, and re-randomize per-episode damage.
    void reset();

    // Advance projectile bookkeeping one control step: hide expired projectiles,
    // refresh the snapshot, and recompute finite-difference velocities.
    void update(float control_dt);

    // Throw at ``target_pos`` (with lead ``target_vel``); Bernoulli-gated.
    void autoThrow(const std::array<float, 3>& target_pos,
                   const std::array<float, 3>& target_vel);
    // Unconditional throw (viewer 'J' key).
    void throwOne(const std::array<float, 3>& target_pos,
                  const std::array<float, 3>& target_vel);

    int count() const;
    const ProjectileSnapshot& snapshot() const;

private:
    struct Impl;
    Impl* p_;
};

}  // namespace ovpp
