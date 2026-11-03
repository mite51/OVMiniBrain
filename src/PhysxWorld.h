// Shared PhysX 5.6.1 world: foundation, physics, dispatcher, scene, materials,
// ground plane, contact reporting, and the fixed-cadence step.
//
// Split out of SmplArticulation so that N humanoid articulations (multi-character
// self-play) plus a projectile pool can live in a single scene and collide with
// one another, matching the IsaacLab training setup. Characters build their links
// into scene()/physics() and register each link for contact reporting; the world's
// contact callback ORs a per-(character, body) touch flag every substep.
#pragma once

#include <cstdint>
#include <vector>

namespace physx {
class PxPhysics;
class PxScene;
class PxMaterial;
}  // namespace physx

namespace ovpp {

class PhysxWorld {
public:
    // enable_ground: add a static ground plane at z = 0 (normal +Z).
    explicit PhysxWorld(bool enable_ground = true);
    ~PhysxWorld();

    PhysxWorld(const PhysxWorld&) = delete;
    PhysxWorld& operator=(const PhysxWorld&) = delete;

    physx::PxPhysics* physics() const;
    physx::PxScene* scene() const;
    physx::PxMaterial* bodyMaterial() const;
    physx::PxMaterial* groundMaterial() const;

    // Advance the shared scene by ``substeps`` iterations of ``dt`` seconds.
    // Contact flags accumulate across the substeps; the caller clears them once
    // per control step (before stepping).
    void step(int substeps, float dt);

    // Contact reporting. Characters register each articulation link with a
    // (character, body) tag; the world's onContact callback sets the flag when the
    // link touches anything (ground, projectile, another character, or itself).
    void ensureContactCapacity(int num_characters, int num_bodies);
    void registerContactLink(void* link, int character, int body);
    void clearContactFlags();
    bool touched(int character, int body) const;

private:
    struct Impl;
    Impl* p_;
};

}  // namespace ovpp
