// Per-tick parity dump writer (NPZ), schema-compatible with
// scripts/app.py::_record_parity_step so scripts/parity_compare.py can diff a
// C++ run against a native ProtoMotions run.
//
// Keys: in_<onnx_input> (feed), out_<onnx_output>, dump_dof_pos/vel, plus
// meta_step_count / meta_motion_file / meta_simulator.
#pragma once

#include "ObsBuilder.h"
#include "OnnxPolicy.h"

#include <map>
#include <string>
#include <vector>

namespace ovpp {

class ParityDump {
public:
    ParityDump(std::string path, std::string motion_file);

    void record(const Feed& feed, const PolicyOutputs& out,
                const std::vector<float>& dof_pos, const std::vector<float>& dof_vel);

    size_t numRecords() const { return num_records_; }
    void write();

private:
    struct Series {
        std::vector<int64_t> per_step_shape;
        std::vector<float> data;  // appended each step
    };

    void append(const std::string& key, const std::vector<int64_t>& per_step_shape,
                const std::vector<float>& data);

    std::string path_;
    std::string motion_file_;
    std::vector<std::string> order_;
    std::map<std::string, Series> series_;
    size_t num_records_ = 0;
};

}  // namespace ovpp
