#include "Config.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <stdexcept>

namespace ovpp {

namespace {

// SMPL canonical schema (config_loader.py fallbacks).
const std::vector<std::string> kSmplBodyNames = {
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
};

const std::vector<std::string> kSmplJoints = {
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
};

std::vector<std::string> smpl_dof_names() {
    std::vector<std::string> out;
    for (const auto& j : kSmplJoints)
        for (const char* ax : {"x", "y", "z"})
            out.push_back(j + "_" + ax);
    return out;
}

// Group fallback gains (regex-fragment, kp, kd) from config_loader.py.
struct GroupGain { std::vector<std::string> tags; float kp, kd; };
const std::vector<GroupGain> kGroupGains = {
    {{"Hip", "Knee", "Ankle"}, 800.0f, 80.0f},
    {{"Toe"}, 500.0f, 50.0f},
    {{"Torso", "Spine", "Chest"}, 1000.0f, 100.0f},
    {{"Neck", "Head"}, 500.0f, 50.0f},
    {{"Thorax", "Shoulder", "Elbow"}, 500.0f, 50.0f},
    {{"Wrist", "Hand"}, 300.0f, 30.0f},
};

void smpl_default_gains(const std::vector<std::string>& dof_names,
                        std::vector<float>& kp, std::vector<float>& kd) {
    kp.clear();
    kd.clear();
    for (const auto& dof : dof_names) {
        std::string joint = dof.substr(0, dof.rfind('_'));
        bool matched = false;
        for (const auto& g : kGroupGains) {
            for (const auto& tag : g.tags) {
                if (joint.find(tag) != std::string::npos) {
                    kp.push_back(g.kp);
                    kd.push_back(g.kd);
                    matched = true;
                    break;
                }
            }
            if (matched) break;
        }
        if (!matched) {
            kp.push_back(500.0f);
            kd.push_back(50.0f);
        }
    }
}

std::vector<float> coerce_float_list(const YAML::Node& node, size_t n,
                                     const std::vector<float>& fallback) {
    if (!node || node.IsNull() || !node.IsSequence())
        return fallback;
    std::vector<float> out;
    for (const auto& v : node) out.push_back(v.as<float>());
    if (out.size() < n)
        out.insert(out.end(), fallback.begin() + out.size(), fallback.end());
    else if (out.size() > n)
        out.resize(n);
    return out;
}

}  // namespace

int Config::substeps() const {
    int n = static_cast<int>(std::lround(control_dt / physics_dt));
    return n < 1 ? 1 : n;
}

const PolicyInputSpec* Config::input_by_key(const std::string& key) const {
    for (const auto& s : policy_inputs)
        if (s.key == key) return &s;
    return nullptr;
}

int Config::body_index(const std::string& name) const {
    for (int i = 0; i < static_cast<int>(body_names.size()); ++i)
        if (body_names[i] == name) return i;
    return -1;
}

Config Config::load(const std::string& yaml_path) {
    Config cfg;
    cfg.yaml_path = yaml_path;
    cfg.body_names = kSmplBodyNames;
    cfg.dof_names = smpl_dof_names();

    YAML::Node blob = YAML::LoadFile(yaml_path);

    auto get_seq_strings = [](const YAML::Node& n) {
        std::vector<std::string> v;
        if (n && n.IsSequence())
            for (const auto& e : n) v.push_back(e.as<std::string>());
        return v;
    };

    cfg.contact_body_names = get_seq_strings(blob["contact_body_names"]);

    // body / joint names (top-level or under robot:).
    YAML::Node body_names = blob["body_names"];
    if (!body_names && blob["robot"]) body_names = blob["robot"]["body_names"];
    YAML::Node joint_names = blob["joint_names"];
    if (!joint_names && blob["robot"]) joint_names = blob["robot"]["joint_names"];
    if (body_names && body_names.IsSequence()) cfg.body_names = get_seq_strings(body_names);
    if (joint_names && joint_names.IsSequence()) cfg.dof_names = get_seq_strings(joint_names);

    const size_t num_dofs = cfg.dof_names.size();

    std::vector<float> def_kp, def_kd;
    smpl_default_gains(cfg.dof_names, def_kp, def_kd);
    std::vector<float> def_eff(num_dofs, 500.0f);
    std::vector<float> def_vel(num_dofs, 100.0f);

    // PD gains: control.* preferred, then top-level default_joint_*.
    YAML::Node control = blob["control"];
    YAML::Node stiffness = control ? control["stiffness"] : YAML::Node();
    if (!stiffness || stiffness.IsNull()) stiffness = blob["default_joint_stiffness"];
    YAML::Node damping = control ? control["damping"] : YAML::Node();
    if (!damping || damping.IsNull()) damping = blob["default_joint_damping"];
    YAML::Node effort = control ? control["effort_limits"] : YAML::Node();
    YAML::Node velocity = control ? control["velocity_limits"] : YAML::Node();

    cfg.pd_stiffness = coerce_float_list(stiffness, num_dofs, def_kp);
    cfg.pd_damping = coerce_float_list(damping, num_dofs, def_kd);
    cfg.pd_effort_limit = coerce_float_list(effort, num_dofs, def_eff);
    cfg.pd_velocity_limit = coerce_float_list(velocity, num_dofs, def_vel);

    // Timing.
    YAML::Node timing = blob["timing"];
    if (timing && timing["control_dt"])
        cfg.control_dt = timing["control_dt"].as<double>();
    else if (blob["dt"])
        cfg.control_dt = blob["dt"].as<double>();
    if (timing && timing["physics_dt"])
        cfg.physics_dt = timing["physics_dt"].as<double>();
    if (timing && timing["decimation"])
        cfg.decimation = timing["decimation"].as<int>();

    // Mimic future indices.
    YAML::Node motion = blob["motion"];
    if (motion && motion["future_step_indices"] && motion["future_step_indices"].IsSequence()) {
        cfg.future_step_indices.clear();
        for (const auto& e : motion["future_step_indices"])
            cfg.future_step_indices.push_back(e.as<int>());
    }

    // policy_inputs / policy_outputs.
    if (blob["policy_inputs"] && blob["policy_inputs"].IsSequence()) {
        for (const auto& e : blob["policy_inputs"]) {
            PolicyInputSpec s;
            s.name = e["name"].as<std::string>();
            s.key = e["key"] ? e["key"].as<std::string>() : s.name;
            if (e["shape"] && e["shape"].IsSequence())
                for (const auto& d : e["shape"]) s.shape.push_back(d.as<int>());
            if (e["kind"] && !e["kind"].IsNull()) s.kind = e["kind"].as<std::string>();
            if (e["output_key"] && !e["output_key"].IsNull())
                s.output_key = e["output_key"].as<std::string>();
            cfg.policy_inputs.push_back(std::move(s));
        }
    }
    if (blob["policy_outputs"] && blob["policy_outputs"].IsSequence()) {
        for (const auto& e : blob["policy_outputs"]) {
            PolicyOutputSpec s;
            s.name = e["name"].as<std::string>();
            s.key = e["key"] ? e["key"].as<std::string>() : s.name;
            if (e["kind"] && !e["kind"].IsNull()) s.kind = e["kind"].as<std::string>();
            if (e["shape"] && e["shape"].IsSequence())
                for (const auto& d : e["shape"]) s.shape.push_back(d.as<int>());
            cfg.policy_outputs.push_back(std::move(s));
        }
    }

    if (cfg.policy_inputs.empty())
        throw std::runtime_error("YAML sidecar has no policy_inputs: " + yaml_path);

    return cfg;
}

}  // namespace ovpp
