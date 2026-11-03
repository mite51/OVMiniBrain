// ONNX Runtime wrapper for the unified-pipeline tracker model.
//
// Port of scripts/policy.py. Loads the ONNX graph (which bakes obs computation
// + actor MLP + action processing), runs one forward pass, and returns the PD
// targets. Linked against onnxruntime.lib; the env's onnxruntime.dll is copied
// next to the exe at build time.
#pragma once

#include "ObsBuilder.h"

#include <string>
#include <vector>

namespace ovpp {

struct PolicyOutputs {
    int rows = 1;  // batch size (== number of characters)
    int dim = 0;   // per-row width (num_dofs)
    std::vector<float> actions;            // [rows * dim]
    std::vector<float> joint_pos_targets;  // [rows * dim]
    bool has_stiffness = false;
    std::vector<float> stiffness_targets;  // [rows * dim]
    bool has_damping = false;
    std::vector<float> damping_targets;    // [rows * dim]

    const float* actionRow(int r) const { return actions.data() + (size_t)r * dim; }
    const float* jointRow(int r) const { return joint_pos_targets.data() + (size_t)r * dim; }
    std::vector<float> jointRowVec(int r) const {
        return {jointRow(r), jointRow(r) + dim};
    }
    std::vector<float> actionRowVec(int r) const {
        return {actionRow(r), actionRow(r) + dim};
    }
};

class OnnxPolicy {
public:
    explicit OnnxPolicy(const std::string& onnx_path);
    ~OnnxPolicy();

    OnnxPolicy(const OnnxPolicy&) = delete;
    OnnxPolicy& operator=(const OnnxPolicy&) = delete;

    PolicyOutputs run(const Feed& feed);

    const std::vector<std::string>& outputNames() const { return output_names_; }

private:
    struct Impl;
    Impl* p_;
    std::vector<std::string> output_names_;
};

}  // namespace ovpp
