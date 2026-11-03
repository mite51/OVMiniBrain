// PhysX 5.6.1 reduced-coordinate articulation built directly from the MJCF.
//
// Replaces scripts/physx_world.py (ovphysx). One instance is one humanoid built
// into a shared PhysxWorld scene (so N characters can collide with one another).
// Constructs the 24-link SMPL articulation (spherical 3-DOF joints, force-mode
// PD, parity caps) and exposes the read/write API the policy loop needs. State is
// reported in kinematic (body_names / dof_names) order, sim coordinates, xyzw
// quats. Physics stepping lives on PhysxWorld, not here.
#pragma once

#include "Config.h"
#include "MjcfModel.h"
#include "PhysxWorld.h"

#include <array>
#include <vector>

namespace ovpp {

struct ArticulationState {
    // All in kinematic order (body_names / dof_names), sim coords, xyzw quats.
    std::vector<std::array<float, 3>> body_pos;       // [B] link actor origin
    std::vector<std::array<float, 4>> body_rot;       // [B] xyzw
    std::vector<std::array<float, 3>> body_lin_vel;   // [B] COM, world
    std::vector<std::array<float, 3>> body_ang_vel;   // [B] world
    std::array<float, 3> root_pos{};
    std::array<float, 4> root_rot{};
    std::array<float, 3> root_lin_vel{};
    std::array<float, 3> root_ang_vel{};
    std::vector<float> dof_pos;  // [D]
    std::vector<float> dof_vel;  // [D]
};

class SmplArticulation {
public:
    SmplArticulation(PhysxWorld& world, const MjcfModel& mjcf, const Config& cfg,
                     int character_index = 0, bool enable_self_collision = true);
    ~SmplArticulation();

    SmplArticulation(const SmplArticulation&) = delete;
    SmplArticulation& operator=(const SmplArticulation&) = delete;

    // Read the full articulation state into ``out`` (resized as needed).
    void readState(ArticulationState& out);

    // Write PD position targets (kinematic dof order, radians).
    void setDofTargets(const std::vector<float>& targets);

    // Set per-DOF PD drive gains (kinematic dof order). Used to apply stamina as
    // ``nominal * stamina`` on reset, matching Simulator.set_joint_gain_scale.
    void setDriveGains(const std::vector<float>& stiffness,
                       const std::vector<float>& damping);

    // Teleport root pose; lin/ang vel optional (nullptr => zero).
    void setRootState(const std::array<float, 3>& pos,
                      const std::array<float, 4>& quat_xyzw,
                      const std::array<float, 3>* lin_vel,
                      const std::array<float, 3>* ang_vel);

    // Teleport joint positions (kinematic order); velocities optional (zeroed).
    void setDofPositions(const std::vector<float>& pos,
                         const std::vector<float>* vel);

    int numBodies() const;
    int numDofs() const;
    int characterIndex() const;

    // Per-link mass in kinematic body order (used for opponent collision-primitive
    // mass). Computed once at construction from the built PhysX links.
    const std::vector<float>& bodyMasses() const;

private:
    struct Impl;
    Impl* p_;
};

}  // namespace ovpp
