#include "ParityDump.h"

#include "npy_npz.hpp"

namespace ovpp {

ParityDump::ParityDump(std::string path, std::string motion_file)
    : path_(std::move(path)), motion_file_(std::move(motion_file)) {}

void ParityDump::append(const std::string& key,
                        const std::vector<int64_t>& per_step_shape,
                        const std::vector<float>& data) {
    auto it = series_.find(key);
    if (it == series_.end()) {
        order_.push_back(key);
        Series s;
        s.per_step_shape = per_step_shape;
        it = series_.emplace(key, std::move(s)).first;
    }
    it->second.data.insert(it->second.data.end(), data.begin(), data.end());
}

void ParityDump::record(const Feed& feed, const PolicyOutputs& out,
                        const std::vector<float>& dof_pos,
                        const std::vector<float>& dof_vel) {
    for (size_t i = 0; i < feed.names.size(); ++i)
        append("in_" + feed.names[i], feed.tensors[i].shape, feed.tensors[i].data);

    // Record character 0's outputs (parity runs use a single character), keeping
    // the historical [1, D] schema regardless of how many rows the batch has.
    const int64_t D = static_cast<int64_t>(out.dim);
    append("out_actions", {1, D}, out.actionRowVec(0));
    append("out_joint_pos_targets", {1, D}, out.jointRowVec(0));
    if (out.has_stiffness)
        append("out_stiffness_targets", {1, D},
               {out.stiffness_targets.begin(), out.stiffness_targets.begin() + D});
    if (out.has_damping)
        append("out_damping_targets", {1, D},
               {out.damping_targets.begin(), out.damping_targets.begin() + D});

    append("dump_dof_pos", {1, (int64_t)dof_pos.size()}, dof_pos);
    append("dump_dof_vel", {1, (int64_t)dof_vel.size()}, dof_vel);

    ++num_records_;
}

void ParityDump::write() {
    if (num_records_ == 0) return;
    std::vector<npz::NpyMember> members;

    for (const auto& key : order_) {
        const Series& s = series_.at(key);
        std::vector<size_t> shape;
        shape.push_back(num_records_);
        for (int64_t d : s.per_step_shape) shape.push_back((size_t)d);
        members.push_back(npz::npy_float32(key, shape, s.data.data(), s.data.size()));
    }

    members.push_back(npz::npy_int64_scalar("meta_step_count", (int64_t)num_records_));
    members.push_back(npz::npy_unicode_scalar("meta_motion_file", motion_file_));
    members.push_back(npz::npy_unicode_scalar("meta_simulator", "physx-5.6.1"));

    npz::write_npz(path_, members);
}

}  // namespace ovpp
