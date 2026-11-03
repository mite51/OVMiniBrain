#include "PhysxWorld.h"

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <memory>
#include <stdexcept>

using namespace physx;

namespace ovpp {

namespace {
constexpr float kContactOffset = 0.02f;
constexpr float kRestOffset = 0.0f;

PxDefaultAllocator gAllocator;
PxDefaultErrorCallback gErrorCallback;

// Per-link tag stored in PxActor::userData so the contact callback can map a
// touching link back to its (character, body) slot.
struct LinkTag {
    int character;
    int body;
};

// Filter shader that enables contact notifications for every colliding pair
// (the default shader reports none). Triggers keep default trigger behavior.
PxFilterFlags contactReportFilter(PxFilterObjectAttributes attr0, PxFilterData,
                                  PxFilterObjectAttributes attr1, PxFilterData,
                                  PxPairFlags& pairFlags, const void*, PxU32) {
    if (PxFilterObjectIsTrigger(attr0) || PxFilterObjectIsTrigger(attr1)) {
        pairFlags = PxPairFlag::eTRIGGER_DEFAULT;
        return PxFilterFlag::eDEFAULT;
    }
    pairFlags = PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eNOTIFY_TOUCH_FOUND |
                PxPairFlag::eNOTIFY_TOUCH_PERSISTS;
    return PxFilterFlag::eDEFAULT;
}

}  // namespace

struct PhysxWorld::Impl : public PxSimulationEventCallback {
    PxFoundation* foundation = nullptr;
    PxPhysics* physics = nullptr;
    PxDefaultCpuDispatcher* dispatcher = nullptr;
    PxScene* scene = nullptr;
    PxMaterial* bodyMaterial = nullptr;
    PxMaterial* groundMaterial = nullptr;
    PxRigidStatic* ground = nullptr;

    int numCharacters = 0;
    int numBodies = 0;
    std::vector<uint8_t> contactFlags;             // [numCharacters * numBodies]
    std::vector<std::unique_ptr<LinkTag>> tags;    // owned userData payloads

    void setTouched(const PxActor* actor) {
        if (!actor || !actor->userData) return;
        const LinkTag* t = static_cast<const LinkTag*>(actor->userData);
        const int idx = t->character * numBodies + t->body;
        if (idx >= 0 && idx < (int)contactFlags.size()) contactFlags[idx] = 1;
    }

    // ---- PxSimulationEventCallback ----
    void onContact(const PxContactPairHeader& header, const PxContactPair*,
                   PxU32) override {
        setTouched(header.actors[0]);
        setTouched(header.actors[1]);
    }
    void onConstraintBreak(PxConstraintInfo*, PxU32) override {}
    void onWake(PxActor**, PxU32) override {}
    void onSleep(PxActor**, PxU32) override {}
    void onTrigger(PxTriggerPair*, PxU32) override {}
    void onAdvance(const PxRigidBody* const*, const PxTransform*, PxU32) override {}

    ~Impl() {
        if (scene) { scene->release(); scene = nullptr; }
        if (dispatcher) { dispatcher->release(); dispatcher = nullptr; }
        if (physics) { PxCloseExtensions(); physics->release(); physics = nullptr; }
        if (foundation) { foundation->release(); foundation = nullptr; }
    }
};

PhysxWorld::PhysxWorld(bool enable_ground) : p_(new Impl()) {
    Impl& s = *p_;

    s.foundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
    if (!s.foundation) throw std::runtime_error("PxCreateFoundation failed");

    PxTolerancesScale scale;
    s.physics = PxCreatePhysics(PX_PHYSICS_VERSION, *s.foundation, scale, false, nullptr);
    if (!s.physics) throw std::runtime_error("PxCreatePhysics failed");
    PxInitExtensions(*s.physics, nullptr);

    PxSceneDesc sceneDesc(s.physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, 0.0f, -9.81f);
    // 0 worker threads -> tasks run on the calling thread (deterministic).
    s.dispatcher = PxDefaultCpuDispatcherCreate(0);
    sceneDesc.cpuDispatcher = s.dispatcher;
    sceneDesc.filterShader = contactReportFilter;
    sceneDesc.simulationEventCallback = &s;
    sceneDesc.solverType = PxSolverType::eTGS;
    sceneDesc.bounceThresholdVelocity = 0.2f;
    s.scene = s.physics->createScene(sceneDesc);

    // Materials. Robot 0.5/0.5, ground 1.0/1.0; default average combine ->
    // foot-ground 0.75. Restitution 0.
    s.bodyMaterial = s.physics->createMaterial(0.5f, 0.5f, 0.0f);
    s.groundMaterial = s.physics->createMaterial(1.0f, 1.0f, 0.0f);

    if (enable_ground) {
        s.ground = PxCreatePlane(*s.physics, PxPlane(0.0f, 0.0f, 1.0f, 0.0f),
                                 *s.groundMaterial);
        PxShape* gshape = nullptr;
        s.ground->getShapes(&gshape, 1);
        if (gshape) {
            gshape->setContactOffset(kContactOffset);
            gshape->setRestOffset(kRestOffset);
        }
        s.scene->addActor(*s.ground);
    }
}

PhysxWorld::~PhysxWorld() { delete p_; }

PxPhysics* PhysxWorld::physics() const { return p_->physics; }
PxScene* PhysxWorld::scene() const { return p_->scene; }
PxMaterial* PhysxWorld::bodyMaterial() const { return p_->bodyMaterial; }
PxMaterial* PhysxWorld::groundMaterial() const { return p_->groundMaterial; }

void PhysxWorld::step(int substeps, float dt) {
    Impl& s = *p_;
    for (int i = 0; i < substeps; ++i) {
        s.scene->simulate(dt);
        s.scene->fetchResults(true);
    }
}

void PhysxWorld::ensureContactCapacity(int num_characters, int num_bodies) {
    Impl& s = *p_;
    s.numCharacters = num_characters;
    s.numBodies = num_bodies;
    s.contactFlags.assign((size_t)num_characters * num_bodies, 0);
}

void PhysxWorld::registerContactLink(void* link, int character, int body) {
    Impl& s = *p_;
    auto tag = std::make_unique<LinkTag>(LinkTag{character, body});
    static_cast<PxActor*>(link)->userData = tag.get();
    s.tags.push_back(std::move(tag));
}

void PhysxWorld::clearContactFlags() {
    std::fill(p_->contactFlags.begin(), p_->contactFlags.end(), 0);
}

bool PhysxWorld::touched(int character, int body) const {
    const Impl& s = *p_;
    const int idx = character * s.numBodies + body;
    if (idx < 0 || idx >= (int)s.contactFlags.size()) return false;
    return s.contactFlags[idx] != 0;
}

}  // namespace ovpp
