#include "OnnxPolicy.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace ovpp {

namespace {
constexpr const char* OUT_ACTIONS = "actions";
constexpr const char* OUT_JOINT_POS = "joint_pos_targets";
constexpr const char* OUT_STIFFNESS = "stiffness_targets";
constexpr const char* OUT_DAMPING = "damping_targets";

std::wstring to_wide(const std::string& s) {
    return std::wstring(s.begin(), s.end());
}
}  // namespace

struct OnnxPolicy::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ovpp"};
    Ort::SessionOptions opts;
    Ort::Session session{nullptr};
    Ort::MemoryInfo mem{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

OnnxPolicy::OnnxPolicy(const std::string& onnx_path) : p_(new Impl()) {
    p_->opts.SetIntraOpNumThreads(1);
    p_->opts.SetInterOpNumThreads(1);
    p_->opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
#ifdef _WIN32
    p_->session = Ort::Session(p_->env, to_wide(onnx_path).c_str(), p_->opts);
#else
    p_->session = Ort::Session(p_->env, onnx_path.c_str(), p_->opts);
#endif

    Ort::AllocatorWithDefaultOptions alloc;
    size_t n_in = p_->session.GetInputCount();
    for (size_t i = 0; i < n_in; ++i) {
        auto name = p_->session.GetInputNameAllocated(i, alloc);
        p_->input_names.emplace_back(name.get());
    }
    size_t n_out = p_->session.GetOutputCount();
    for (size_t i = 0; i < n_out; ++i) {
        auto name = p_->session.GetOutputNameAllocated(i, alloc);
        p_->output_names.emplace_back(name.get());
    }
    output_names_ = p_->output_names;

    auto has = [&](const char* n) {
        return std::find(p_->output_names.begin(), p_->output_names.end(), n) !=
               p_->output_names.end();
    };
    if (!has(OUT_ACTIONS) || !has(OUT_JOINT_POS))
        throw std::runtime_error(
            "ONNX model missing required outputs 'actions'/'joint_pos_targets'");
}

OnnxPolicy::~OnnxPolicy() { delete p_; }

PolicyOutputs OnnxPolicy::run(const Feed& feed) {
    Impl& s = *p_;

    // Build input tensors in the order the session expects. Bool inputs (e.g.
    // body_contacts) are carried as 0/1 floats in the feed and materialized into
    // a bool buffer here; those buffers must outlive the Run call.
    std::vector<Ort::Value> in_values;
    std::vector<const char*> in_names;
    std::vector<std::unique_ptr<bool[]>> bool_buffers;
    in_values.reserve(s.input_names.size());
    in_names.reserve(s.input_names.size());

    for (const auto& want : s.input_names) {
        // Find matching feed entry.
        int idx = -1;
        for (size_t i = 0; i < feed.names.size(); ++i)
            if (feed.names[i] == want) { idx = (int)i; break; }
        if (idx < 0)
            throw std::runtime_error("ONNX feed missing input '" + want + "'");
        const Tensor& t = feed.tensors[idx];
        if (t.is_bool) {
            auto b = std::make_unique<bool[]>(t.data.size());
            for (size_t i = 0; i < t.data.size(); ++i) b[i] = (t.data[i] != 0.0f);
            in_values.push_back(Ort::Value::CreateTensor<bool>(
                s.mem, b.get(), t.data.size(), t.shape.data(), t.shape.size()));
            bool_buffers.push_back(std::move(b));
        } else {
            in_values.push_back(Ort::Value::CreateTensor<float>(
                s.mem, const_cast<float*>(t.data.data()), t.data.size(),
                t.shape.data(), t.shape.size()));
        }
        in_names.push_back(want.c_str());
    }

    // Request outputs: actions, joint_pos_targets, then stiffness/damping if present.
    auto has = [&](const char* n) {
        return std::find(s.output_names.begin(), s.output_names.end(), n) !=
               s.output_names.end();
    };
    std::vector<const char*> out_names = {OUT_ACTIONS, OUT_JOINT_POS};
    bool want_stiff = has(OUT_STIFFNESS);
    bool want_damp = has(OUT_DAMPING);
    if (want_stiff) out_names.push_back(OUT_STIFFNESS);
    if (want_damp) out_names.push_back(OUT_DAMPING);

    auto outs = s.session.Run(Ort::RunOptions{nullptr}, in_names.data(),
                              in_values.data(), in_values.size(), out_names.data(),
                              out_names.size());

    auto to_vec = [](Ort::Value& v) {
        auto info = v.GetTensorTypeAndShapeInfo();
        size_t n = info.GetElementCount();
        const float* d = v.GetTensorData<float>();
        return std::vector<float>(d, d + n);
    };
    auto rows_of = [](Ort::Value& v) {
        auto shape = v.GetTensorTypeAndShapeInfo().GetShape();
        return shape.empty() ? 1 : (int)shape[0];
    };

    PolicyOutputs result;
    result.actions = to_vec(outs[0]);
    result.joint_pos_targets = to_vec(outs[1]);
    result.rows = rows_of(outs[0]);
    result.dim = result.rows > 0 ? (int)(result.actions.size() / result.rows) : 0;
    size_t next = 2;
    if (want_stiff) {
        result.has_stiffness = true;
        result.stiffness_targets = to_vec(outs[next++]);
    }
    if (want_damp) {
        result.has_damping = true;
        result.damping_targets = to_vec(outs[next++]);
    }
    return result;
}

}  // namespace ovpp
