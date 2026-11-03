#include "ProjectilePool.h"

#include "PhysxWorld.h"

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

using namespace physx;

namespace ovpp {

namespace {
constexpr float kContactOffset = 0.02f;
constexpr float kRestOffset = 0.0f;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

float lerp(const std::array<float, 2>& r, int i, int n) {
    if (n <= 1) return r[0];
    return r[0] + (r[1] - r[0]) * (float)i / (float)(n - 1);
}
}  // namespace

struct ProjectilePool::Impl {
    PhysxWorld* world = nullptr;
    ProjectileParams params;
    std::mt19937_64 rng;

    int P = 0;
    std::vector<PxRigidDynamic*> bodies;
    std::vector<float> throw_time;   // per slot; kNegInf when hidden
    float sim_time = 0.0f;

    // Static per-slot geometry / obs encoding.
    std::vector<float> radius, extent_z, mass;
    std::vector<std::array<float, 2>> shape;
    std::vector<std::string> shape_type;

    // Per-slot damage (randomized per episode).
    std::vector<float> damage;

    // Finite-difference bookkeeping.
    std::vector<std::array<float, 3>> prev_pos;
    std::vector<float> prev_active;

    ProjectileSnapshot snap;

    float uni(float lo, float hi) {
        std::uniform_real_distribution<float> d(lo, hi);
        return d(rng);
    }
    float nrm() {
        std::normal_distribution<float> d(0.0f, 1.0f);
        return d(rng);
    }

    void setEnabled(int i, bool enabled) {
        bodies[i]->setActorFlag(PxActorFlag::eDISABLE_SIMULATION, !enabled);
    }

    void hide(int i) {
        // Zero motion only while still simulating (setting velocity on a
        // simulation-disabled actor is not allowed); pose teleport is fine either
        // way. throw_time == kNegInf means the slot is already hidden/disabled.
        if (throw_time[i] != kNegInf) {
            bodies[i]->setLinearVelocity(PxVec3(0));
            bodies[i]->setAngularVelocity(PxVec3(0));
        }
        bodies[i]->setGlobalPose(PxTransform(PxVec3(0, 0, params.hide_z)));
        setEnabled(i, false);
        throw_time[i] = kNegInf;
    }

    void randomizeDamage() {
        const int n = P;
        float frac = uni(params.dangerous_fraction_range[0],
                         params.dangerous_fraction_range[1]);
        int count = (int)std::lround(frac * n);
        std::vector<int> order(n);
        for (int i = 0; i < n; ++i) order[i] = i;
        std::shuffle(order.begin(), order.end(), rng);
        for (int i = 0; i < n; ++i) damage[i] = params.baseline_damage;
        for (int k = 0; k < count && k < n; ++k)
            damage[order[k]] = uni(params.damage_range[0], params.damage_range[1]);
    }
};

ProjectilePool::ProjectilePool(PhysxWorld& world, const ProjectileParams& params,
                               uint64_t seed)
    : p_(new Impl()) {
    Impl& s = *p_;
    s.world = &world;
    s.params = params;
    s.rng.seed(seed);
    s.P = params.num_projectiles;

    PxPhysics* physics = world.physics();
    PxMaterial* mat = world.bodyMaterial();

    const auto& shapes = params.shapes.empty()
                             ? std::vector<std::string>{"box"}
                             : params.shapes;

    s.bodies.resize(s.P, nullptr);
    s.throw_time.assign(s.P, kNegInf);
    s.radius.assign(s.P, 0.0f);
    s.extent_z.assign(s.P, 0.0f);
    s.mass.assign(s.P, 0.0f);
    s.shape.assign(s.P, {0.0f, 0.0f});
    s.shape_type.assign(s.P, "box");
    s.damage.assign(s.P, params.baseline_damage);
    s.prev_pos.assign(s.P, {0, 0, params.hide_z});
    s.prev_active.assign(s.P, 0.0f);

    const float density = params.density;
    for (int i = 0; i < s.P; ++i) {
        float sz = lerp(params.cube_half_size_range, i, s.P);  // characteristic half-size
        const std::string& st = shapes[i % shapes.size()];
        PxGeometryHolder geom;
        float vol = 0.0f;
        if (st == "box") {
            geom.storeAny(PxBoxGeometry(sz, sz, sz));
            s.radius[i] = 0.0f;
            s.extent_z[i] = 2.0f * sz;
            s.shape[i] = {1.0f, 0.0f};
            vol = s.extent_z[i] * s.extent_z[i] * s.extent_z[i];
        } else if (st == "sphere") {
            geom.storeAny(PxSphereGeometry(sz));
            s.radius[i] = sz;
            s.extent_z[i] = 0.0f;
            s.shape[i] = {0.0f, 1.0f};
            vol = 4.0f / 3.0f * PxPi * sz * sz * sz;
        } else {  // capsule
            geom.storeAny(PxCapsuleGeometry(sz, sz));  // radius sz, halfHeight sz (len 2sz)
            s.radius[i] = sz;
            s.extent_z[i] = 2.0f * sz;
            s.shape[i] = {0.0f, 0.0f};
            vol = PxPi * sz * sz * (2.0f * sz) + 4.0f / 3.0f * PxPi * sz * sz * sz;
        }
        s.shape_type[i] = st;
        s.mass[i] = vol * density;

        PxRigidDynamic* body = physics->createRigidDynamic(
            PxTransform(PxVec3(0, 0, params.hide_z)));
        PxShape* shp = physics->createShape(geom.any(), *mat, true);
        // PhysX capsule axis is local +X, which is fine (orientation is randomized
        // on throw); no re-orientation needed here.
        shp->setContactOffset(kContactOffset);
        shp->setRestOffset(kRestOffset);
        body->attachShape(*shp);
        shp->release();
        PxRigidBodyExt::updateMassAndInertia(*body, density);
        body->setActorFlag(PxActorFlag::eDISABLE_SIMULATION, true);
        world.scene()->addActor(*body);
        s.bodies[i] = body;
    }

    // Snapshot static arrays.
    s.snap.P = s.P;
    s.snap.radius = s.radius;
    s.snap.extent_z = s.extent_z;
    s.snap.shape = s.shape;
    s.snap.mass = s.mass;
    s.snap.damage = s.damage;
    s.snap.pos.assign(s.P, {0, 0, params.hide_z});
    s.snap.rot.assign(s.P, {0, 0, 0, 1});
    s.snap.lin_vel.assign(s.P, {0, 0, 0});
    s.snap.active.assign(s.P, 0.0f);

    s.randomizeDamage();
    s.snap.damage = s.damage;
}

ProjectilePool::~ProjectilePool() {
    if (p_) {
        for (PxRigidDynamic* b : p_->bodies)
            if (b) { p_->world->scene()->removeActor(*b); b->release(); }
        delete p_;
        p_ = nullptr;
    }
}

int ProjectilePool::count() const { return p_->P; }
const ProjectileSnapshot& ProjectilePool::snapshot() const { return p_->snap; }

void ProjectilePool::reset() {
    Impl& s = *p_;
    s.sim_time = 0.0f;
    for (int i = 0; i < s.P; ++i) {
        s.hide(i);
        s.prev_pos[i] = {0, 0, s.params.hide_z};
        s.prev_active[i] = 0.0f;
    }
    s.randomizeDamage();
    s.snap.damage = s.damage;
    for (int i = 0; i < s.P; ++i) {
        s.snap.pos[i] = {0, 0, s.params.hide_z};
        s.snap.rot[i] = {0, 0, 0, 1};
        s.snap.lin_vel[i] = {0, 0, 0};
        s.snap.active[i] = 0.0f;
    }
}

void ProjectilePool::throwOne(const std::array<float, 3>& target_pos,
                              const std::array<float, 3>& target_vel) {
    Impl& s = *p_;
    if (s.P == 0) return;
    // Prefer a free (hidden) slot: score = rand + 2 * is_free, argmax.
    int slot = 0;
    float best = -1e30f;
    for (int i = 0; i < s.P; ++i) {
        float free = (s.throw_time[i] == kNegInf) ? 1.0f : 0.0f;
        float score = s.uni(0.0f, 1.0f) + 2.0f * free;
        if (score > best) { best = score; slot = i; }
    }

    float angle = s.uni(0.0f, 2.0f * PxPi);
    float dist = s.uni(s.params.spawn_distance_range[0], s.params.spawn_distance_range[1]);
    float h = s.uni(s.params.spawn_height_range[0], s.params.spawn_height_range[1]);
    PxVec3 spawn(target_pos[0] + std::cos(angle) * dist,
                 target_pos[1] + std::sin(angle) * dist, target_pos[2] + h);

    PxVec3 dir(target_pos[0] - spawn.x, target_pos[1] - spawn.y, target_pos[2] - spawn.z);
    dir += PxVec3(s.nrm(), s.nrm(), s.nrm()) * s.params.direction_noise_std;
    if (dir.magnitudeSquared() < 1e-8f) dir = PxVec3(0, 0, -1);
    dir.normalize();
    float speed = s.uni(s.params.speed_range[0], s.params.speed_range[1]);
    PxVec3 vel = dir * speed;
    vel.x += target_vel[0];
    vel.y += target_vel[1];

    PxQuat rot(s.nrm(), s.nrm(), s.nrm(), s.nrm());
    if (rot.magnitudeSquared() < 1e-8f) rot = PxQuat(PxIdentity);
    rot.normalize();
    PxVec3 angvel(s.nrm() * 4.0f, s.nrm() * 4.0f, s.nrm() * 4.0f);

    PxRigidDynamic* body = s.bodies[slot];
    s.setEnabled(slot, true);
    body->setGlobalPose(PxTransform(spawn, rot));
    body->setLinearVelocity(vel);
    body->setAngularVelocity(angvel);
    s.throw_time[slot] = s.sim_time;
    // Zero the finite-difference velocity on the first active step (avoid the
    // spike from the hidden->thrown teleport).
    s.prev_pos[slot] = {spawn.x, spawn.y, spawn.z};
    s.prev_active[slot] = 0.0f;
}

void ProjectilePool::autoThrow(const std::array<float, 3>& target_pos,
                               const std::array<float, 3>& target_vel) {
    Impl& s = *p_;
    if (s.params.auto_throw_prob <= 0.0f) return;
    if (s.uni(0.0f, 1.0f) < s.params.auto_throw_prob) throwOne(target_pos, target_vel);
}

void ProjectilePool::update(float control_dt) {
    Impl& s = *p_;
    s.sim_time += control_dt;

    for (int i = 0; i < s.P; ++i) {
        // Expire.
        if (s.throw_time[i] != kNegInf &&
            (s.sim_time - s.throw_time[i]) > s.params.hide_delay) {
            s.hide(i);
        }

        bool active = (s.throw_time[i] != kNegInf);
        PxTransform pose = s.bodies[i]->getGlobalPose();
        std::array<float, 3> pos = {pose.p.x, pose.p.y, pose.p.z};

        // Finite-difference velocity (zero unless active on both this and the
        // previous step).
        std::array<float, 3> vel = {0, 0, 0};
        if (active && s.prev_active[i] > 0.5f && control_dt > 0.0f) {
            vel = {(pos[0] - s.prev_pos[i][0]) / control_dt,
                   (pos[1] - s.prev_pos[i][1]) / control_dt,
                   (pos[2] - s.prev_pos[i][2]) / control_dt};
        }

        s.snap.pos[i] = pos;
        s.snap.rot[i] = {pose.q.x, pose.q.y, pose.q.z, pose.q.w};
        s.snap.lin_vel[i] = vel;
        s.snap.active[i] = active ? 1.0f : 0.0f;

        s.prev_pos[i] = pos;
        s.prev_active[i] = active ? 1.0f : 0.0f;
    }
    s.snap.damage = s.damage;
}

}  // namespace ovpp
