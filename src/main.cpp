// OVMiniBrain C++ inference harness (PhysX 5.6.1 SDK).
//
// Mirrors protomotions/inference_agent.py's fixed-cadence policy loop for the
// motion-tracker and fighting-mimic contracts: read PhysX state -> build the
// ONNX feed (current state + mimic future + action history, plus collision
// primitives / body contacts / stamina for the fight model) -> ONNX forward pass
// -> push raw actions to history -> write joint_pos_targets -> step physics by
// ceil(control_dt/physics_dt) substeps.
//
// Supports N policy-driven characters sharing one PhysxWorld (batched through a
// single dynamic-batch ONNX Run), a thrown-projectile pool, per-episode stamina
// randomization, and smooth reference re-anchoring. Optional NPZ parity dump
// (PPP_PARITY_DUMP_PATH / PPP_PARITY_MAX_STEPS) records character 0.

#include "CollisionPrimitives.h"
#include "Config.h"
#include "Evaluation.h"
#include "MjcfModel.h"
#include "MotionCache.h"
#include "ObsBuilder.h"
#include "OnnxPolicy.h"
#include "ParityDump.h"
#include "PhysxWorld.h"
#include "ProjectilePool.h"
#include "Realign.h"
#include "Renderer.h"
#include "SmplArticulation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ovpp;

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct Args {
    std::string eval_output;
    double eval_threshold = 0.5;
    std::string onnx;
    std::string yaml;
    std::string motion_cache;
    std::string mjcf = "C:/Git/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml";
    long max_steps = 0;       // 0 = run to motion end
    bool self_collision = true;
    int tracking_log_every = 30;
    double control_dt = -1.0;
    double physics_dt = -1.0;
    bool render = false;
    int width = 1280;
    int height = 720;
    double speed = 1.0;

    // Multi-character + fight knobs.
    int num_characters = 1;
    float spawn_radius = 1.0f;
    uint64_t seed = 0;
    bool projectiles = true;                 // enabled iff the model is a fight model
    int num_projectiles = 3;
    float auto_throw_prob = 0.01f;
    std::array<float, 2> projectile_speed = {12.0f, 22.0f};
    float projectile_density = 300.0f;
    bool randomize_stamina = false;
    std::array<float, 2> stamina_range = {0.2f, 1.0f};
};

std::string arg_value(int argc, char** argv, int& i) {
    if (i + 1 >= argc) { std::cerr << "Missing value for " << argv[i] << "\n"; std::exit(2); }
    return argv[++i];
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--eval-output") a.eval_output = arg_value(argc, argv, i);
        else if (k == "--eval-threshold") a.eval_threshold = std::stod(arg_value(argc, argv, i));
        else if (k == "--onnx") a.onnx = arg_value(argc, argv, i);
        else if (k == "--yaml") a.yaml = arg_value(argc, argv, i);
        else if (k == "--motion-cache") a.motion_cache = arg_value(argc, argv, i);
        else if (k == "--mjcf") a.mjcf = arg_value(argc, argv, i);
        else if (k == "--max-steps") a.max_steps = std::stol(arg_value(argc, argv, i));
        else if (k == "--no-self-collision") a.self_collision = false;
        else if (k == "--tracking-log-every") a.tracking_log_every = std::stoi(arg_value(argc, argv, i));
        else if (k == "--control-dt") a.control_dt = std::stod(arg_value(argc, argv, i));
        else if (k == "--physics-dt") a.physics_dt = std::stod(arg_value(argc, argv, i));
        else if (k == "--render") a.render = true;
        else if (k == "--width") a.width = std::stoi(arg_value(argc, argv, i));
        else if (k == "--height") a.height = std::stoi(arg_value(argc, argv, i));
        else if (k == "--speed") a.speed = std::stod(arg_value(argc, argv, i));
        else if (k == "--num-characters") a.num_characters = std::stoi(arg_value(argc, argv, i));
        else if (k == "--spawn-radius") a.spawn_radius = std::stof(arg_value(argc, argv, i));
        else if (k == "--seed") a.seed = std::stoull(arg_value(argc, argv, i));
        else if (k == "--no-projectiles") a.projectiles = false;
        else if (k == "--num-projectiles") a.num_projectiles = std::stoi(arg_value(argc, argv, i));
        else if (k == "--auto-throw-prob") a.auto_throw_prob = std::stof(arg_value(argc, argv, i));
        else if (k == "--projectile-density") a.projectile_density = std::stof(arg_value(argc, argv, i));
        else if (k == "--projectile-speed") {
            a.projectile_speed[0] = std::stof(arg_value(argc, argv, i));
            a.projectile_speed[1] = std::stof(arg_value(argc, argv, i));
        } else if (k == "--randomize-stamina") a.randomize_stamina = true;
        else if (k == "--no-randomize-stamina") a.randomize_stamina = false;
        else if (k == "--stamina-range") {
            a.randomize_stamina = true;
            a.stamina_range[0] = std::stof(arg_value(argc, argv, i));
            a.stamina_range[1] = std::stof(arg_value(argc, argv, i));
        } else if (k == "-h" || k == "--help") {
            std::cout <<
                "Usage: ovminibrain --onnx model.onnx --yaml sidecar.yaml "
                "--motion-cache clip.pmc [--mjcf smpl_humanoid.xml]\n"
                "  Core:   [--max-steps N] [--no-self-collision] [--tracking-log-every N]\n"
                "          [--control-dt S] [--physics-dt S]\n"
                "  Evaluate: [--eval-output report.json] [--eval-threshold METRES] (full clip, healthy, fixed reference)\n"
                "  Viewer: [--render] [--width W] [--height H] [--speed X]  (J throws)\n"
                "  Multi:  [--num-characters N] [--spawn-radius R] [--seed S]\n"
                "  Throw:  [--no-projectiles] [--num-projectiles K] [--auto-throw-prob P]\n"
                "          [--projectile-speed LO HI] [--projectile-density D]\n"
                "  Stamina:[--randomize-stamina] [--no-randomize-stamina] [--stamina-range LO HI]\n"
                "  Env:    PPP_PARITY_DUMP_PATH, PPP_PARITY_MAX_STEPS\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown arg: " << k << "\n";
            std::exit(2);
        }
    }
    if (a.onnx.empty() || a.motion_cache.empty()) {
        std::cerr << "--onnx and --motion-cache are required (see --help).\n";
        std::exit(2);
    }
    if (a.yaml.empty()) {
        std::string stem = a.onnx;
        auto dot = stem.rfind('.');
        if (dot != std::string::npos) stem = stem.substr(0, dot);
        a.yaml = stem + ".yaml";
    }
    if (a.num_characters < 1) a.num_characters = 1;
    if (!a.eval_output.empty()) {
        a.num_characters=1; a.projectiles=false; a.randomize_stamina=false;
        a.seed=0;
    }
    return a;
}

// ---- reference placement (yaw + XY translate) -----------------------------
// Relocates the shared clip for one character: rotate about the clip's frame-0
// root anchor by ``yaw``, translate so the anchor lands on ``pivot``, then add
// the smooth realign ``offset`` (XY). N==1 is identity (parity path).
struct Placement {
    float cx = 1.0f, sx = 0.0f;      // cos/sin yaw
    float qzs = 0.0f, qzc = 1.0f;    // yaw half-quat (sin, cos)
    std::array<float, 2> anchor{{0, 0}};
    std::array<float, 2> pivot{{0, 0}};
    std::array<float, 2> offset{{0, 0}};
};

void setYaw(Placement& p, float yaw) {
    p.cx = std::cos(yaw);
    p.sx = std::sin(yaw);
    p.qzs = std::sin(0.5f * yaw);
    p.qzc = std::cos(0.5f * yaw);
}

std::array<float, 3> placePos(const Placement& p, const float* q, bool with_offset) {
    float dx = q[0] - p.anchor[0], dy = q[1] - p.anchor[1];
    float rx = p.cx * dx - p.sx * dy, ry = p.sx * dx + p.cx * dy;
    std::array<float, 3> o = {p.pivot[0] + rx, p.pivot[1] + ry, q[2]};
    if (with_offset) { o[0] += p.offset[0]; o[1] += p.offset[1]; }
    return o;
}

std::array<float, 4> placeRot(const Placement& p, const float* q) {  // q xyzw
    const float ax = 0, ay = 0, az = p.qzs, aw = p.qzc;
    const float bx = q[0], by = q[1], bz = q[2], bw = q[3];
    return {aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz};
}

std::array<float, 3> placeVec(const Placement& p, const float* v) {
    return {p.cx * v[0] - p.sx * v[1], p.sx * v[0] + p.cx * v[1], v[2]};
}

bool endsWith(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// Opponent key bodies (regex [Head, Pelvis, .*_Shoulder, .*_Elbow, .*_Knee,
// .*_Ankle]), ascending body index.
std::vector<int> keyBodyIds(const Config& cfg) {
    std::vector<int> ids;
    for (int i = 0; i < cfg.num_bodies(); ++i) {
        const std::string& n = cfg.body_names[i];
        if (n == "Head" || n == "Pelvis" || endsWith(n, "_Shoulder") ||
            endsWith(n, "_Elbow") || endsWith(n, "_Knee") || endsWith(n, "_Ankle"))
            ids.push_back(i);
    }
    return ids;
}

// Frozen contact-body order (must match training exactly).
std::vector<int> contactBodyIds(const Config& cfg) {
    static const char* kOrder[] = {"L_Ankle", "L_Toe",  "R_Ankle", "R_Toe",
                                   "L_Hand",  "R_Hand", "Head",    "Torso"};
    std::vector<int> ids;
    std::vector<std::string> names=cfg.contact_body_names;
    if(names.empty())for(const char* n:kOrder)names.emplace_back(n);
    for (const auto& name : names) {
        int idx = cfg.body_index(name);
        if (idx < 0) throw std::runtime_error(std::string("contact body '") + name +
                                              "' not found in body_names");
        ids.push_back(idx);
    }
    return ids;
}

void staminaGains(const Config& cfg, const std::vector<float>& stamina,
                  std::vector<float>& k, std::vector<float>& d) {
    const int D = cfg.num_dofs();
    const int S = (int)stamina.size();
    const int dpb = (S > 0) ? D / S : 1;
    k.resize(D);
    d.resize(D);
    for (int i = 0; i < D; ++i) {
        int bi = (S > 0) ? std::min(i / dpb, S - 1) : 0;
        float sc = stamina[bi];
        k[i] = cfg.pd_stiffness[i] * sc;
        d[i] = cfg.pd_damping[i] * sc;
    }
}

}  // namespace

int main(int argc, char** argv) try {
    Args args = parse_args(argc, argv);
    const bool evaluating = !args.eval_output.empty();
    Evaluation evaluation(args.eval_threshold);

    std::cout << "[ovpp] Loading config: " << args.yaml << "\n";
    Config cfg = Config::load(args.yaml);
    if (args.control_dt > 0.0) {
        std::cout << "[ovpp] override control_dt " << cfg.control_dt << " -> "
                  << args.control_dt << "\n";
        cfg.control_dt = args.control_dt;
    }
    if (args.physics_dt > 0.0) {
        std::cout << "[ovpp] override physics_dt " << cfg.physics_dt << " -> "
                  << args.physics_dt << "\n";
        cfg.physics_dt = args.physics_dt;
    }

    const bool fight = cfg.is_fight_model();
    const int N = args.num_characters;
    std::cout << "[ovpp] " << cfg.num_bodies() << " bodies, " << cfg.num_dofs()
              << " dofs, control_dt=" << cfg.control_dt
              << " physics_dt=" << cfg.physics_dt << " substeps=" << cfg.substeps()
              << " future=" << cfg.future_steps()
              << (cfg.has_input_key("current.dof_pos") ? " [hierarchical mini-brain]" : (fight ? " [FIGHT model]" : " [tracker model]"))
              << " num_characters=" << N << "\n";
    if (fight)
        std::cout << "[ovpp]   fight inputs: M=" << cfg.num_collision_primitives()
                  << " contacts=" << cfg.num_contact_bodies()
                  << " stamina=" << cfg.num_stamina_bodies() << "\n";
    if (N > 1 && !fight)
        std::cerr << "[warn] --num-characters " << N
                  << " with a non-fight model: characters will not observe each other.\n";

    std::cout << "[ovpp] Loading MJCF: " << args.mjcf << "\n";
    MjcfModel mjcf = MjcfModel::load(args.mjcf);
    if ((int)mjcf.bodies.size() != cfg.num_bodies()) {
        std::cerr << "MJCF body count " << mjcf.bodies.size() << " != config "
                  << cfg.num_bodies() << "\n";
        return 1;
    }

    std::cout << "[ovpp] Loading motion cache: " << args.motion_cache << "\n";
    MotionCache motion = MotionCache::load(args.motion_cache);
    if (motion.B != cfg.num_bodies() || motion.D != cfg.num_dofs()) {
        std::cerr << "Motion cache shape (B=" << motion.B << ",D=" << motion.D
                  << ") != config (B=" << cfg.num_bodies() << ",D=" << cfg.num_dofs()
                  << ")\n";
        return 1;
    }
    const float dt = motion.control_dt;
    if(evaluating && (motion.length<=0 || !std::isfinite(motion.length) || std::abs(dt-cfg.control_dt)>1e-6 || std::abs(cfg.substeps()*cfg.physics_dt-cfg.control_dt)>1e-6))
        throw std::runtime_error("Evaluation requires positive duration and matching cache/control/physics timing");
    if (std::abs((double)dt - cfg.control_dt) > 1e-6)
        std::cerr << "[warn] motion cache control_dt=" << dt
                  << " != config control_dt=" << cfg.control_dt
                  << " (using cache value for ticks)\n";

    if(evaluating && cfg.future_step_indices.empty())throw std::runtime_error("Evaluation requires future step indices");
    const int B = cfg.num_bodies();
    const int D = cfg.num_dofs();
    const int F = cfg.future_steps();
    if(evaluating) {
        if(motion.K < (int)std::ceil(motion.length / dt)+*std::max_element(cfg.future_step_indices.begin(),cfg.future_step_indices.end())+1)
            throw std::runtime_error("Evaluation cache does not cover the full clip and future lookahead");
        for(const auto* values : {&motion.body_pos,&motion.body_rot,&motion.body_vel,&motion.body_ang_vel,&motion.dof_pos,&motion.dof_vel})
            for(float value:*values)if(!std::isfinite(value))throw std::runtime_error("Non-finite evaluation reference");
    }
    const int substeps = cfg.substeps();
    const float physics_dt = (float)cfg.physics_dt;

    // ---- world + characters -------------------------------------------------
    std::cout << "[ovpp] Building PhysX world + " << N << " character(s) (self_collision="
              << (args.self_collision ? "on" : "off") << ")\n";
    PhysxWorld world(/*enable_ground=*/true);
    std::vector<std::unique_ptr<SmplArticulation>> chars;
    for (int c = 0; c < N; ++c)
        chars.push_back(std::make_unique<SmplArticulation>(world, mjcf, cfg, c,
                                                           args.self_collision));
    world.ensureContactCapacity(N, B);

    // Placements: N==1 identity; otherwise spawn on a circle facing the center.
    const float* p0 = motion.bodyPos(0);
    std::array<float, 2> anchor = {p0[0], p0[1]};
    std::vector<Placement> place(N);
    for (int c = 0; c < N; ++c) {
        place[c].anchor = anchor;
        if (N == 1) {
            place[c].pivot = anchor;
            setYaw(place[c], 0.0f);
        } else {
            float a = 2.0f * kPi * (float)c / (float)N;
            place[c].pivot = {args.spawn_radius * std::cos(a),
                              args.spawn_radius * std::sin(a)};
            setYaw(place[c], a + kPi);  // face inward
        }
    }

    // ---- ONNX + obs ---------------------------------------------------------
    std::cout << "[ovpp] Loading ONNX policy: " << args.onnx << "\n";
    OnnxPolicy policy(args.onnx);
    ObsBuilder obs(cfg, N);

    // ---- fight subsystems ---------------------------------------------------
    std::mt19937_64 rng(args.seed);
    std::unique_ptr<ProjectilePool> projectiles;
    std::vector<int> key_ids, contact_ids;
    CollisionSceneInfo scene;
    const int S = cfg.num_stamina_bodies();
    std::vector<std::vector<float>> stamina(N, std::vector<float>(S, 1.0f));
    RealignParams realign_params;
    const bool use_projectiles = fight && args.projectiles && args.num_projectiles > 0;

    if (cfg.num_contact_bodies() > 0) {
        contact_ids = contactBodyIds(cfg);
        if ((int)contact_ids.size() != cfg.num_contact_bodies())
            throw std::runtime_error("Contact body names do not match policy input width");
    }
    if (fight) {
        key_ids = keyBodyIds(cfg);
        scene.ground_z = 0.0f;
        scene.opponent_key_body_ids = key_ids;
        scene.body_mass = chars[0]->bodyMasses();
        if (use_projectiles) {
            ProjectileParams pp;
            pp.num_projectiles = args.num_projectiles;
            pp.density = args.projectile_density;
            pp.speed_range = args.projectile_speed;
            pp.auto_throw_prob = args.auto_throw_prob;
            projectiles = std::make_unique<ProjectilePool>(world, pp, args.seed ^ 0x9E3779B9u);
        }
        std::cout << "[ovpp]   projectiles=" << (use_projectiles ? args.num_projectiles : 0)
                  << " auto_throw_prob=" << (use_projectiles ? args.auto_throw_prob : 0.0f)
                  << " randomize_stamina=" << (args.randomize_stamina ? "on" : "off") << "\n";
    }

    // Parity dump (optional; records character 0).
    std::unique_ptr<ParityDump> parity;
    long parity_max_steps = 0;
    if (const char* pth = std::getenv("PPP_PARITY_DUMP_PATH")) {
        parity = std::make_unique<ParityDump>(pth, args.motion_cache);
        if (const char* m = std::getenv("PPP_PARITY_MAX_STEPS"))
            parity_max_steps = std::stol(m);
        std::cout << "[ovpp] Parity dump -> " << pth
                  << " (max_steps=" << parity_max_steps << ")\n";
    }

    std::unique_ptr<Renderer> renderer;
    if (args.render) {
        std::cout << "[ovpp] Opening viewer (" << args.width << "x" << args.height
                  << ", speed=" << args.speed << "x). Space=pause, Right=step, "
                     "R=reset cam, J=throw, drag=orbit, scroll=zoom, Esc=quit.\n";
        renderer = std::make_unique<Renderer>(args.width, args.height, mjcf);
    }

    // ---- per-character reset ------------------------------------------------
    std::uniform_real_distribution<float> stamina_dist(args.stamina_range[0],
                                                       args.stamina_range[1]);
    auto resetChar = [&](int c) {
        const float* r0 = motion.bodyRot(0);
        const float* v0 = motion.bodyVel(0);
        const float* d0 = motion.dofPos(0);
        place[c].offset = {0.0f, 0.0f};
        std::array<float, 3> root_pos = placePos(place[c], motion.bodyPos(0), false);
        root_pos[2] += cfg.ref_respawn_offset_z;
        std::array<float, 4> root_rot = placeRot(place[c], r0);
        std::array<float, 3> root_lin = placeVec(place[c], v0);
        std::vector<float> dof0(d0, d0 + D);
        std::array<float,3> root_ang=placeVec(place[c],motion.bodyAngVel(0));
        std::vector<float> dof_vel(motion.dofVel(0),motion.dofVel(0)+D);
        chars[c]->setRootState(root_pos, root_rot, &root_lin, evaluating ? &root_ang : nullptr);
        chars[c]->setDofPositions(dof0, evaluating ? &dof_vel : nullptr);
        chars[c]->setDofTargets(dof0);

        // Stamina (per episode) -> PhysX drive gains + body_stamina obs.
        if (S > 0) {
            for (int i = 0; i < S; ++i)
                stamina[c][i] = args.randomize_stamina ? stamina_dist(rng) : 1.0f;
            std::vector<float> ks, ds;
            staminaGains(cfg, stamina[c], ks, ds);
            chars[c]->setDriveGains(ks, ds);
        }
        obs.resetHistory(c);
    };

    auto resetAll = [&]() {
        for (int c = 0; c < N; ++c) resetChar(c);
        if (projectiles) projectiles->reset();
    };
    resetAll();

    // ---- per-character scratch ----------------------------------------------
    std::vector<ArticulationState> states(N);
    std::vector<MotionFutureData> futures(N);
    std::vector<CollisionPrimitives> prims(N);
    std::vector<std::vector<uint8_t>> contacts(N, std::vector<uint8_t>(contact_ids.size(), 0));
    for (int c = 0; c < N; ++c) {
        futures[c].dof_pos.resize((size_t)F * D);
        futures[c].dof_vel.resize((size_t)F * D);
        futures[c].F = F;
        futures[c].B = B;
        futures[c].pos.resize((size_t)F * B * 3);
        futures[c].rot.resize((size_t)F * B * 4);
        futures[c].vel.resize((size_t)F * B * 3);
        futures[c].ang_vel.resize((size_t)F * B * 3);
    }

    long step_count = 0;
    long total_step_count = 0;
    double track_sum = 0.0;
    long track_n = 0;
    const int max_motion_tick = motion.K - 1;

    // Render buffers.
    std::vector<std::vector<std::array<float, 3>>> refPos(N, std::vector<std::array<float, 3>>(B));
    std::vector<std::vector<std::array<float, 4>>> refRot(N, std::vector<std::array<float, 4>>(B));
    using clk = std::chrono::steady_clock;
    auto next_frame = clk::now();

    std::cout << "[ovpp] Starting policy loop (motion length=" << motion.length << "s, "
              << (int)std::ceil(motion.length / dt) << " ticks).\n";

    bool running = true;
    while (running) {
        const int tick = (int)step_count;

        // 1) Read all characters' states first (opponents observe each other).
        for (int c = 0; c < N; ++c) chars[c]->readState(states[c]);

        // 2) Realign offset update + reference ghosts.
        for (int c = 0; c < N; ++c) {
            const float* rp = motion.bodyPos(tick);
            const float* rv = motion.bodyVel(tick);
            std::array<float, 3> refRoot = placePos(place[c], rp, false);
            std::array<float, 3> refRootVel = placeVec(place[c], rv);
            if (!evaluating) place[c].offset = computeSmoothRealignOffset(
                place[c].offset,
                {states[c].root_pos[0], states[c].root_pos[1]},
                {states[c].root_lin_vel[0], states[c].root_lin_vel[1]},
                {refRoot[0], refRoot[1]}, {refRootVel[0], refRootVel[1]},
                realign_params, dt);
        }

        // 3) Build per-character mimic future (placed + realigned XY).
        for (int c = 0; c < N; ++c) {
            for (int fi = 0; fi < F; ++fi) {
                int src = tick + cfg.future_step_indices[fi];
                // Local joint coordinates are invariant under root placement.
                std::copy_n(motion.dofPos(src), D, futures[c].dof_pos.begin() + (size_t)fi * D);
                std::copy_n(motion.dofVel(src), D, futures[c].dof_vel.begin() + (size_t)fi * D);
                const float* fp = motion.bodyPos(src);
                const float* fr = motion.bodyRot(src);
                const float* fv = motion.bodyVel(src);
                const float* fa = motion.bodyAngVel(src);
                for (int b = 0; b < B; ++b) {
                    auto pp = placePos(place[c], fp + b * 3, true);
                    auto rr = placeRot(place[c], fr + b * 4);
                    auto vv = placeVec(place[c], fv + b * 3);
                    auto aa = placeVec(place[c], fa + b * 3);
                    size_t o3 = ((size_t)fi * B + b) * 3;
                    size_t o4 = ((size_t)fi * B + b) * 4;
                    std::copy(pp.begin(), pp.end(), futures[c].pos.begin() + o3);
                    std::copy(rr.begin(), rr.end(), futures[c].rot.begin() + o4);
                    std::copy(vv.begin(), vv.end(), futures[c].vel.begin() + o3);
                    std::copy(aa.begin(), aa.end(), futures[c].ang_vel.begin() + o3);
                }
            }
        }

        // 4) Fight extras (collision primitives, contacts) per character.
        const ProjectileSnapshot* snap = projectiles ? &projectiles->snapshot() : nullptr;
        if (fight) {
            for (int c = 0; c < N; ++c) {
                std::vector<const ArticulationState*> opponents;
                for (int o = 0; o < N; ++o)
                    if (o != c) opponents.push_back(&states[o]);
                buildCollisionPrimitives(prims[c], cfg.num_collision_primitives(),
                                         states[c], scene.ground_z, snap, opponents,
                                         scene);
                for (size_t i = 0; i < contact_ids.size(); ++i)
                    contacts[c][i] = world.touched(c, contact_ids[i]) ? 1 : 0;
            }
        }

        // 5) Assemble the batched feed and run one ONNX pass for all N.
        std::vector<CharacterObsInputs> inputs(N);
        for (int c = 0; c < N; ++c) {
            inputs[c].state = &states[c];
            inputs[c].future = &futures[c];
            inputs[c].ground_height = 0.0f;
            for (size_t i = 0; i < contact_ids.size(); ++i)
                contacts[c][i] = world.touched(c, contact_ids[i]) ? 1 : 0;
            inputs[c].body_contacts = &contacts[c];
            inputs[c].body_stamina = &stamina[c];
            if (fight) {
                inputs[c].primitives = &prims[c];
                inputs[c].body_contacts = &contacts[c];
                inputs[c].body_stamina = &stamina[c];
            }
        }
        Feed feed = obs.build(inputs);
        PolicyOutputs out = policy.run(feed);

        for (int c = 0; c < N; ++c) {
            for(float v:out.actionRowVec(c))if(!std::isfinite(v))throw std::runtime_error("Non-finite policy action");
            for(float v:out.jointRowVec(c))if(!std::isfinite(v))throw std::runtime_error("Non-finite policy target");
            obs.pushAction(c, out.actionRowVec(c));
            chars[c]->setDofTargets(out.jointRowVec(c));
        }

        if (parity) parity->record(feed, out, states[0].dof_pos, states[0].dof_vel);

        // 6) Viewer (draw pre-step state), pacing, and manual throw.
        if (renderer) {
            for (int c = 0; c < N; ++c) {
                const float* rp = motion.bodyPos(tick);
                const float* rr = motion.bodyRot(tick);
                for (int b = 0; b < B; ++b) {
                    refPos[c][b] = placePos(place[c], rp + b * 3, true);
                    refRot[c][b] = placeRot(place[c], rr + b * 4);
                }
            }
            std::vector<const ArticulationState*> sims(N);
            for (int c = 0; c < N; ++c) sims[c] = &states[c];
            do {
                renderer->drawFrame(sims, refPos, refRot, snap);
                if (renderer->throwRequested() && projectiles)
                    projectiles->throwOne(states[0].root_pos, states[0].root_lin_vel);
                if (renderer->shouldClose()) { running = false; break; }
            } while (renderer->paused() && !renderer->stepRequested());
            if (!running) break;
            if (args.speed > 0.0) {
                std::this_thread::sleep_until(next_frame);
                auto now = clk::now();
                auto period = std::chrono::duration<double>((double)dt / args.speed);
                if (next_frame < now) next_frame = now;
                next_frame += std::chrono::duration_cast<clk::duration>(period);
            }
        }

        // 7) Auto-throw (targets character 0), then step physics.
        if (projectiles)
            projectiles->autoThrow(states[0].root_pos, states[0].root_lin_vel);
        world.clearContactFlags();
        world.step(substeps, physics_dt);
        if (projectiles) projectiles->update(dt);
        if(evaluating) {
            chars[0]->readState(states[0]);
            evaluation.add(states[0],motion,tick+1,(tick+1)*(double)dt);
        }

        // Tracking error (character 0) vs the placed+realigned ref at this tick.
        {
            const float* refp = motion.bodyPos(evaluating ? tick+1 : tick);
            double err = 0.0;
            for (int b = 0; b < B; ++b) {
                auto rp = placePos(place[0], refp + b * 3, true);
                double dx = states[0].body_pos[b][0] - rp[0];
                double dy = states[0].body_pos[b][1] - rp[1];
                double dz = states[0].body_pos[b][2] - rp[2];
                err += std::sqrt(dx * dx + dy * dy + dz * dz);
            }
            track_sum += err / B;
            ++track_n;
        }

        ++step_count;
        ++total_step_count;
        double t_next = step_count * (double)dt;
        if (args.tracking_log_every > 0 && track_n >= args.tracking_log_every) {
            std::cout << "[ovpp] tracking err (char0, mean per-body L2, last " << track_n
                      << "): " << (track_sum / track_n) << " m  [t=" << t_next << "/"
                      << motion.length << "]\n";
            track_sum = 0.0;
            track_n = 0;
        }

        if (parity && parity_max_steps > 0 && (long)parity->numRecords() >= parity_max_steps) {
            std::cout << "[ovpp] Reached PPP_PARITY_MAX_STEPS=" << parity_max_steps
                      << ", stopping.\n";
            running = false;
        } else if (args.max_steps > 0 && total_step_count >= args.max_steps) {
            std::cout << "[ovpp] Reached --max-steps=" << args.max_steps << ", stopping.\n";
            running = false;
        } else if (t_next >= motion.length || tick >= max_motion_tick) {
            std::cout << "[ovpp] Motion finished at step " << step_count << ".\n";
            if (evaluating || parity || (!args.render && args.max_steps <= 0)) running = false;  // one headless episode by default
            else { resetAll(); step_count = 0; }
        }
    }

    if(evaluating) evaluation.write(args.eval_output,motion,args.onnx,args.motion_cache,total_step_count*(double)dt);
    if (parity) {
        parity->write();
        std::cout << "[ovpp] Wrote parity dump (" << parity->numRecords() << " records).\n";
    }
    std::cout << "[ovpp] Done.\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "[ovpp] FATAL: " << e.what() << "\n";
    return 1;
}
