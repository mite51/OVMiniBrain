#include "SmplArticulation.h"

#include <PxPhysicsAPI.h>

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

using namespace physx;

namespace ovpp {

namespace {

// 1000 deg/s in rad/s — the per-body maxAngularVelocity native IsaacLab
// authors (the USD schema value is in deg/s; the SDK setter wants rad/s).
constexpr float kMaxAngVel = 17.453293f;     // 1000 deg/s
constexpr float kMaxLinVel = 1000.0f;
constexpr float kMaxDepenVel = 1.0f;
constexpr float kJointMaxForce = 500.0f;     // effort_limit
constexpr float kJointMaxVel = 100.0f;       // velocity_limit (rad/s)
constexpr float kContactOffset = 0.02f;
constexpr float kRestOffset = 0.0f;

PxQuat quat_from_to(const PxVec3& from, const PxVec3& to) {
    PxVec3 a = from.getNormalized();
    PxVec3 b = to.getNormalized();
    float d = a.dot(b);
    if (d >= 1.0f - 1e-6f) return PxQuat(PxIdentity);
    if (d <= -1.0f + 1e-6f) {
        // 180 deg: pick any axis perpendicular to ``a``.
        PxVec3 axis = PxVec3(1, 0, 0).cross(a);
        if (axis.magnitudeSquared() < 1e-6f) axis = PxVec3(0, 1, 0).cross(a);
        return PxQuat(PxPi, axis.getNormalized());
    }
    PxVec3 c = a.cross(b);
    PxQuat q(c.x, c.y, c.z, 1.0f + d);
    return q.getNormalized();
}

}  // namespace

struct SmplArticulation::Impl {
    // Borrowed from the shared PhysxWorld (not owned).
    PhysxWorld* world = nullptr;
    PxPhysics* physics = nullptr;
    PxScene* scene = nullptr;
    PxMaterial* bodyMaterial = nullptr;

    PxArticulationReducedCoordinate* art = nullptr;
    PxArticulationCache* cache = nullptr;

    int numBodies = 0;
    int numDofs = 0;
    int characterIndex = 0;

    std::vector<PxArticulationLink*> bodyLinks;  // kinematic body order
    std::vector<PxU32> bodyLowLevel;             // low-level link index per body
    std::vector<int> kinDofToCache;              // kinematic dof -> cache dof index
    std::vector<float> bodyMasses;               // kinematic body order

    // kinematic dof -> (axis enum, PhysX joint) for runtime gain updates.
    std::vector<PxArticulationJointReducedCoordinate*> dofJoint;
    std::vector<PxArticulationAxis::Enum> dofAxis;

    ~Impl() {
        if (cache) { cache->release(); cache = nullptr; }
        if (art && scene) { scene->removeArticulation(*art); }
        if (art) { art->release(); art = nullptr; }
    }

    void buildShape(PxArticulationLink* link, const Geom& geom) {
        PxShape* shape = nullptr;
        if (geom.type == GeomType::Box) {
            PxBoxGeometry box(static_cast<float>(geom.size[0]),
                              static_cast<float>(geom.size[1]),
                              static_cast<float>(geom.size[2]));
            shape = physics->createShape(box, *bodyMaterial, true);
            PxQuat q(static_cast<float>(geom.quat[1]),  // x
                     static_cast<float>(geom.quat[2]),  // y
                     static_cast<float>(geom.quat[3]),  // z
                     static_cast<float>(geom.quat[0]));  // w
            shape->setLocalPose(PxTransform(
                PxVec3(static_cast<float>(geom.pos[0]),
                       static_cast<float>(geom.pos[1]),
                       static_cast<float>(geom.pos[2])),
                q.getNormalized()));
        } else {
            PxVec3 a(static_cast<float>(geom.fromto_a[0]),
                     static_cast<float>(geom.fromto_a[1]),
                     static_cast<float>(geom.fromto_a[2]));
            PxVec3 b(static_cast<float>(geom.fromto_b[0]),
                     static_cast<float>(geom.fromto_b[1]),
                     static_cast<float>(geom.fromto_b[2]));
            PxVec3 seg = b - a;
            float halfHeight = 0.5f * seg.magnitude();
            PxCapsuleGeometry cap(static_cast<float>(geom.radius), halfHeight);
            shape = physics->createShape(cap, *bodyMaterial, true);
            // PhysX capsule axis is local +X; align it to the segment.
            PxQuat q = quat_from_to(PxVec3(1, 0, 0), seg);
            PxVec3 center = 0.5f * (a + b);
            shape->setLocalPose(PxTransform(center, q));
        }
        shape->setContactOffset(kContactOffset);
        shape->setRestOffset(kRestOffset);
        link->attachShape(*shape);
        shape->release();  // link retains a reference
    }

    void configureLinkBody(PxArticulationLink* link) {
        link->setMaxAngularVelocity(kMaxAngVel);
        link->setMaxLinearVelocity(kMaxLinVel);
        link->setMaxDepenetrationVelocity(kMaxDepenVel);
        link->setAngularDamping(0.0f);
        link->setLinearDamping(0.0f);
        link->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES, true);
        link->setRigidBodyFlag(PxRigidBodyFlag::eRETAIN_ACCELERATIONS, false);
    }
};

SmplArticulation::SmplArticulation(PhysxWorld& world, const MjcfModel& mjcf,
                                   const Config& cfg, int character_index,
                                   bool enable_self_collision)
    : p_(new Impl()) {
    Impl& s = *p_;
    s.world = &world;
    s.physics = world.physics();
    s.scene = world.scene();
    s.bodyMaterial = world.bodyMaterial();
    s.numBodies = static_cast<int>(mjcf.bodies.size());
    s.numDofs = cfg.num_dofs();
    s.characterIndex = character_index;
    s.dofJoint.assign(s.numDofs, nullptr);
    s.dofAxis.assign(s.numDofs, PxArticulationAxis::eTWIST);

    // Articulation.
    s.art = s.physics->createArticulationReducedCoordinate();
    s.art->setArticulationFlag(PxArticulationFlag::eFIX_BASE, false);
    s.art->setArticulationFlag(PxArticulationFlag::eDISABLE_SELF_COLLISION,
                               !enable_self_collision);
    s.art->setSolverIterationCounts(4, 4);
    s.art->setSleepThreshold(0.0f);

    // Cumulative rest global positions (all rest rotations identity).
    std::vector<PxVec3> restPos(s.numBodies, PxVec3(0));
    s.bodyLinks.assign(s.numBodies, nullptr);

    for (int b = 0; b < s.numBodies; ++b) {
        const MjcfBody& body = mjcf.bodies[b];
        PxVec3 offset(static_cast<float>(body.pos[0]),
                      static_cast<float>(body.pos[1]),
                      static_cast<float>(body.pos[2]));
        PxArticulationLink* parentLink = nullptr;
        if (body.parent >= 0) {
            restPos[b] = restPos[body.parent] + offset;
            parentLink = s.bodyLinks[body.parent];
        } else {
            restPos[b] = offset;  // root rest pose origin (overwritten at reset)
        }

        PxArticulationLink* link =
            s.art->createLink(parentLink, PxTransform(restPos[b], PxQuat(PxIdentity)));
        s.bodyLinks[b] = link;

        s.buildShape(link, body.geom);
        PxRigidBodyExt::updateMassAndInertia(*link, static_cast<float>(body.geom.density));
        s.configureLinkBody(link);

        if (body.parent >= 0) {
            // Non-root: configure the inbound spherical joint.
            PxArticulationJointReducedCoordinate* joint = link->getInboundJoint();
            joint->setJointType(PxArticulationJointType::eSPHERICAL);
            joint->setParentPose(PxTransform(offset, PxQuat(PxIdentity)));
            joint->setChildPose(PxTransform(PxVec3(0), PxQuat(PxIdentity)));

            const int jointIdx = b - 1;  // body order matches dof joint order
            const PxArticulationAxis::Enum axes[3] = {
                PxArticulationAxis::eTWIST,    // rotX
                PxArticulationAxis::eSWING1,   // rotY
                PxArticulationAxis::eSWING2,   // rotZ
            };
            for (int a = 0; a < 3; ++a) {
                int d = jointIdx * 3 + a;
                joint->setMotion(axes[a], PxArticulationMotion::eFREE);
                joint->setDriveParams(
                    axes[a],
                    PxArticulationDrive(cfg.pd_stiffness[d], cfg.pd_damping[d],
                                        kJointMaxForce, PxArticulationDriveType::eFORCE));
                joint->setArmature(axes[a], static_cast<float>(body.armature));
                joint->setMaxJointVelocity(axes[a], kJointMaxVel);
                s.dofJoint[d] = joint;   // for runtime (stamina) gain updates
                s.dofAxis[d] = axes[a];
            }
        }
    }

    s.scene->addArticulation(*s.art);

    // Register each link for contact reporting and cache its mass (opponent
    // collision-primitive slots use per-body mass). The caller sizes the world's
    // contact-flag buffer once via PhysxWorld::ensureContactCapacity(N, numBodies).
    s.bodyMasses.assign(s.numBodies, 1.0f);
    for (int b = 0; b < s.numBodies; ++b) {
        s.world->registerContactLink(s.bodyLinks[b], character_index, b);
        s.bodyMasses[b] = s.bodyLinks[b]->getMass();
    }

    if (static_cast<int>(s.art->getDofs()) != s.numDofs) {
        throw std::runtime_error(
            "Articulation DOF mismatch: built " +
            std::to_string(s.art->getDofs()) + " expected " +
            std::to_string(s.numDofs));
    }

    s.cache = s.art->createCache();

    // Low-level link index per body, and cache DOF start offset per link.
    s.bodyLowLevel.assign(s.numBodies, 0);
    std::map<PxU32, PxArticulationLink*> byLowLevel;
    for (int b = 0; b < s.numBodies; ++b) {
        PxU32 ll = s.bodyLinks[b]->getLinkIndex();
        s.bodyLowLevel[b] = ll;
        byLowLevel[ll] = s.bodyLinks[b];
    }
    // dofStart[lowLevelIndex] following ascending low-level order.
    std::map<PxU32, int> dofStartByLowLevel;
    int acc = 0;
    for (auto& kv : byLowLevel) {
        dofStartByLowLevel[kv.first] = acc;
        acc += static_cast<int>(kv.second->getInboundJointDof());
    }

    // kinematic dof -> cache dof index.
    s.kinDofToCache.assign(s.numDofs, 0);
    for (int b = 1; b < s.numBodies; ++b) {
        PxU32 ll = s.bodyLowLevel[b];
        int start = dofStartByLowLevel[ll];
        int jointIdx = b - 1;
        for (int a = 0; a < 3; ++a)
            s.kinDofToCache[jointIdx * 3 + a] = start + a;  // twist,swing1,swing2
    }
}

SmplArticulation::~SmplArticulation() {
    delete p_;
    p_ = nullptr;
}

int SmplArticulation::numBodies() const { return p_->numBodies; }
int SmplArticulation::numDofs() const { return p_->numDofs; }
int SmplArticulation::characterIndex() const { return p_->characterIndex; }
const std::vector<float>& SmplArticulation::bodyMasses() const { return p_->bodyMasses; }

void SmplArticulation::setDriveGains(const std::vector<float>& stiffness,
                                     const std::vector<float>& damping) {
    Impl& s = *p_;
    for (int d = 0; d < s.numDofs; ++d) {
        PxArticulationJointReducedCoordinate* joint = s.dofJoint[d];
        if (!joint) continue;
        joint->setDriveParams(
            s.dofAxis[d],
            PxArticulationDrive(stiffness[d], damping[d], kJointMaxForce,
                                PxArticulationDriveType::eFORCE));
    }
}

void SmplArticulation::readState(ArticulationState& out) {
    Impl& s = *p_;
    const int B = s.numBodies;
    const int D = s.numDofs;
    out.body_pos.resize(B);
    out.body_rot.resize(B);
    out.body_lin_vel.resize(B);
    out.body_ang_vel.resize(B);
    out.dof_pos.resize(D);
    out.dof_vel.resize(D);

    s.art->copyInternalStateToCache(
        *s.cache, PxArticulationCacheFlag::ePOSITION |
                      PxArticulationCacheFlag::eVELOCITY |
                      PxArticulationCacheFlag::eLINK_VELOCITY);

    for (int b = 0; b < B; ++b) {
        PxTransform gp = s.bodyLinks[b]->getGlobalPose();
        out.body_pos[b] = {gp.p.x, gp.p.y, gp.p.z};
        out.body_rot[b] = {gp.q.x, gp.q.y, gp.q.z, gp.q.w};
        const PxSpatialVelocity& v = s.cache->linkVelocity[s.bodyLowLevel[b]];
        out.body_lin_vel[b] = {v.linear.x, v.linear.y, v.linear.z};
        out.body_ang_vel[b] = {v.angular.x, v.angular.y, v.angular.z};
    }

    for (int d = 0; d < D; ++d) {
        int idx = s.kinDofToCache[d];
        out.dof_pos[d] = s.cache->jointPosition[idx];
        out.dof_vel[d] = s.cache->jointVelocity[idx];
    }

    out.root_pos = out.body_pos[0];
    out.root_rot = out.body_rot[0];
    out.root_lin_vel = out.body_lin_vel[0];
    out.root_ang_vel = out.body_ang_vel[0];
}

void SmplArticulation::setDofTargets(const std::vector<float>& targets) {
    Impl& s = *p_;
    for (int d = 0; d < s.numDofs; ++d)
        s.cache->jointTargetPositions[s.kinDofToCache[d]] = targets[d];
    s.art->applyCache(*s.cache, PxArticulationCacheFlag::eJOINT_TARGET_POSITIONS);
}

void SmplArticulation::setRootState(const std::array<float, 3>& pos,
                                    const std::array<float, 4>& quat_xyzw,
                                    const std::array<float, 3>* lin_vel,
                                    const std::array<float, 3>* ang_vel) {
    Impl& s = *p_;
    PxArticulationRootLinkData& root = *s.cache->rootLinkData;
    root.transform = PxTransform(
        PxVec3(pos[0], pos[1], pos[2]),
        PxQuat(quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]).getNormalized());
    root.worldLinVel = lin_vel ? PxVec3((*lin_vel)[0], (*lin_vel)[1], (*lin_vel)[2])
                               : PxVec3(0);
    root.worldAngVel = ang_vel ? PxVec3((*ang_vel)[0], (*ang_vel)[1], (*ang_vel)[2])
                               : PxVec3(0);
    s.art->applyCache(*s.cache, PxArticulationCacheFlag::eROOT_TRANSFORM |
                                    PxArticulationCacheFlag::eROOT_VELOCITIES);
}

void SmplArticulation::setDofPositions(const std::vector<float>& pos,
                                       const std::vector<float>* vel) {
    Impl& s = *p_;
    for (int d = 0; d < s.numDofs; ++d) {
        int idx = s.kinDofToCache[d];
        s.cache->jointPosition[idx] = pos[d];
        s.cache->jointVelocity[idx] = vel ? (*vel)[d] : 0.0f;
    }
    s.art->applyCache(*s.cache, PxArticulationCacheFlag::ePOSITION |
                                    PxArticulationCacheFlag::eVELOCITY);
}

}  // namespace ovpp
