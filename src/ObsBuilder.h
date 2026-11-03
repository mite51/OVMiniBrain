// Assemble the ONNX feed for one policy tick from current sim state, the
// mimic future, an action history ring buffer, the ground height, and (for the
// fighting-mimic model) collision primitives, per-body contacts, and stamina.
//
// Port of scripts/obs_builder.py, extended for the fight contract. The input
// contract is driven entirely by the YAML ``policy_inputs`` (dotted keys), so
// this stays forward-compatible with exports that add/drop inputs. All N
// characters are stacked into a single batched feed (leading dim == N); the
// exported graph has a dynamic batch axis, so one Session::Run drives them all.
#pragma once

#include "CollisionPrimitives.h"
#include "Config.h"
#include "SmplArticulation.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ovpp {

// Mimic future reference, flat row-major. F = number of future_step_indices.
struct MotionFutureData {
    int F = 0, B = 0;
    std::vector<float> dof_pos;  // [F,D], local joint angles (not world transformed)
    std::vector<float> dof_vel;  // [F,D]
    std::vector<float> pos;      // [F,B,3]
    std::vector<float> rot;      // [F,B,4] xyzw
    std::vector<float> vel;      // [F,B,3]
    std::vector<float> ang_vel;  // [F,B,3]
};

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
    // ONNX bool inputs (e.g. body_contacts) are still carried as 0.0/1.0 floats
    // here so the parity dump and stacking stay uniform; OnnxPolicy materializes
    // a bool buffer at Run time when this flag is set.
    bool is_bool = false;
};

// Ordered feed (parallel arrays) ready for onnxruntime Run.
struct Feed {
    std::vector<std::string> names;
    std::vector<Tensor> tensors;
};

// Per-character inputs for one tick. The fight extras are consulted only when
// the loaded model declares the matching inputs.
struct CharacterObsInputs {
    const ArticulationState* state = nullptr;
    const MotionFutureData* future = nullptr;
    float ground_height = 0.0f;
    const CollisionPrimitives* primitives = nullptr;      // M slots (fight only)
    const std::vector<uint8_t>* body_contacts = nullptr;  // [numContactBodies]
    const std::vector<float>* body_stamina = nullptr;     // [numStaminaBodies]
};

class ObsBuilder {
public:
    ObsBuilder(const Config& cfg, int num_characters);

    void resetHistory();                 // all characters
    void resetHistory(int character);    // one character
    // Push one character's raw action (len == num_dofs) into its history ring.
    void pushAction(int character, const std::vector<float>& actions_raw);

    // Build the batched feed (leading dim == num_characters).
    Feed build(const std::vector<CharacterObsInputs>& chars);

    int historySteps() const { return history_steps_; }
    int numCharacters() const { return num_characters_; }

private:
    const Config& cfg_;
    int num_characters_;
    int num_bodies_;
    int num_dofs_;
    int history_steps_;
    // [num_characters * max(1,H) * D], slot 0 == newest per character.
    std::vector<float> history_buf_;

    int histStride() const { return (history_steps_ > 0 ? history_steps_ : 1) * num_dofs_; }
};

}  // namespace ovpp
