// Resolved inference config parsed from the unified-pipeline YAML sidecar.
//
// Port of scripts/config_loader.py::ResolvedConfig. Carries the joint / body
// schema, per-DOF PD gains, control timing, future look-ahead, and the ONNX
// input/output specs that drive ObsBuilder + OnnxPolicy.
#pragma once

#include <string>
#include <vector>
#include <optional>

namespace ovpp {

// One entry from the YAML ``policy_inputs`` block.
struct PolicyInputSpec {
    std::string name;          // ONNX input name, e.g. "current_rigid_body_pos"
    std::string key;           // dotted context key, e.g. "current.rigid_body_pos"
    std::vector<int> shape;    // full shape incl. leading batch dim
    std::string kind;          // optional
    std::string output_key;    // optional (historical inputs)
};

struct PolicyOutputSpec {
    std::string name;
    std::string key;
    std::string kind;
    std::vector<int> shape;
};

class Config {
public:
    // Joint / body schema.
    std::vector<std::string> contact_body_names;
    std::vector<std::string> body_names;   // 24
    std::vector<std::string> dof_names;    // 69

    // Per-DOF PD gains (len == num_dofs).
    std::vector<float> pd_stiffness;
    std::vector<float> pd_damping;
    std::vector<float> pd_effort_limit;    // defaults to 500 each
    std::vector<float> pd_velocity_limit;  // defaults to 100 each

    // Drive token: "force" (matches IsaacLab's effective PD). The C++ port
    // always uses PhysX force-mode drives.
    std::string drive_type = "force";

    // Spawn lift (not in YAML); ProtoMotions' canonical 5 cm clearance.
    float ref_respawn_offset_z = 0.05f;

    // Control timing.
    double control_dt = 1.0 / 30.0;
    double physics_dt = 1.0 / 120.0;
    int decimation = 4;

    // Mimic future look-ahead (units of control_dt).
    std::vector<int> future_step_indices = {1};

    // ONNX schema.
    std::vector<PolicyInputSpec> policy_inputs;
    std::vector<PolicyOutputSpec> policy_outputs;

    std::string yaml_path;

    int num_bodies() const { return static_cast<int>(body_names.size()); }
    int num_dofs() const { return static_cast<int>(dof_names.size()); }
    int future_steps() const { return static_cast<int>(future_step_indices.size()); }
    double policy_fps() const { return 1.0 / control_dt; }

    // ceil(control_dt / physics_dt), matching PhysxWorld.step's substep count.
    int substeps() const;

    const PolicyInputSpec* input_by_key(const std::string& key) const;
    bool has_input_key(const std::string& key) const { return input_by_key(key) != nullptr; }

    // Fighting-mimic input sizes, read straight from the sidecar shapes so the
    // client never hardcodes them. Each returns 0 when the corresponding input is
    // absent (i.e. a stock tracker export without the fight observations).

    // Body index of ``name`` in ``body_names`` (-1 if absent).
    int body_index(const std::string& name) const;

    // M = collision-primitive candidate-buffer capacity (``collision_primitives.pos``
    // second dim). 0 when the model has no collision-primitive obs.
    int num_collision_primitives() const {
        const PolicyInputSpec* s = input_by_key("collision_primitives.pos");
        return (s && s->shape.size() >= 2) ? s->shape[1] : 0;
    }

    // Per-actuated-body stamina vector length (``body_stamina`` second dim).
    int num_stamina_bodies() const {
        const PolicyInputSpec* s = input_by_key("body_stamina");
        return (s && s->shape.size() >= 2) ? s->shape[1] : 0;
    }

    // Frozen contact-body subset width (``body_contacts`` second dim).
    int num_contact_bodies() const {
        const PolicyInputSpec* s = input_by_key("body_contacts");
        return (s && s->shape.size() >= 2) ? s->shape[1] : 0;
    }

    bool is_fight_model() const { return num_collision_primitives() > 0; }

    // Parse a unified-pipeline YAML sidecar. Throws std::runtime_error on
    // failure. Mirrors load_config_from_yaml in config_loader.py.
    static Config load(const std::string& yaml_path);
};

}  // namespace ovpp
