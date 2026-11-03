#include "ObsBuilder.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace ovpp {

namespace {
constexpr const char* KEY_POS = "current.rigid_body_pos";
constexpr const char* KEY_ROT = "current.rigid_body_rot";
constexpr const char* KEY_VEL = "current.rigid_body_vel";
constexpr const char* KEY_ANG = "current.rigid_body_ang_vel";
constexpr const char* KEY_GROUND = "ground_heights";
constexpr const char* KEY_HIST = "historical.actions";
constexpr const char* KEY_FUT_POS = "mimic.future_pos";
constexpr const char* KEY_FUT_ROT = "mimic.future_rot";
constexpr const char* KEY_FUT_VEL = "mimic.future_vel";
constexpr const char* KEY_FUT_ANG = "mimic.future_ang_vel";
// Hierarchical controller inputs; analytical load features stay inside ONNX.
constexpr const char* KEY_DOF_POS = "current.dof_pos";
constexpr const char* KEY_DOF_VEL = "current.dof_vel";
constexpr const char* KEY_FUT_DOF_POS = "mimic.future_dof_pos";
constexpr const char* KEY_FUT_DOF_VEL = "mimic.future_dof_vel";
// Fighting-mimic keys.
constexpr const char* KEY_CONTACTS = "body_contacts";
constexpr const char* KEY_STAMINA = "body_stamina";
constexpr const char* KEY_CP_POS = "collision_primitives.pos";
constexpr const char* KEY_CP_ROT = "collision_primitives.rot";
constexpr const char* KEY_CP_VEL = "collision_primitives.lin_vel";
constexpr const char* KEY_CP_RADIUS = "collision_primitives.radius";
constexpr const char* KEY_CP_EXTENT = "collision_primitives.extent_z";
constexpr const char* KEY_CP_DAMAGE = "collision_primitives.damage";
constexpr const char* KEY_CP_MASS = "collision_primitives.mass";
constexpr const char* KEY_CP_SHAPE = "collision_primitives.shape";
constexpr const char* KEY_CP_VALID = "collision_primitives.valid";

bool isKnownKey(const std::string& k) {
    static const char* known[] = {
        KEY_POS, KEY_ROT, KEY_VEL, KEY_ANG, KEY_GROUND, KEY_HIST,
        KEY_FUT_POS, KEY_FUT_ROT, KEY_FUT_VEL, KEY_FUT_ANG,
        KEY_CONTACTS, KEY_STAMINA, KEY_CP_POS, KEY_CP_ROT, KEY_CP_VEL,
        KEY_CP_RADIUS, KEY_CP_EXTENT, KEY_CP_DAMAGE, KEY_CP_MASS, KEY_CP_SHAPE,
        KEY_CP_VALID, KEY_DOF_POS, KEY_DOF_VEL, KEY_FUT_DOF_POS, KEY_FUT_DOF_VEL,
    };
    for (const char* n : known)
        if (k == n) return true;
    return false;
}
}  // namespace

ObsBuilder::ObsBuilder(const Config& cfg, int num_characters)
    : cfg_(cfg),
      num_characters_(num_characters > 0 ? num_characters : 1),
      num_bodies_(cfg.num_bodies()),
      num_dofs_(cfg.num_dofs()),
      history_steps_(0) {
    const PolicyInputSpec* hist = cfg.input_by_key(KEY_HIST);
    if (hist && hist->shape.size() >= 3) history_steps_ = hist->shape[1];
    history_buf_.assign((size_t)num_characters_ * histStride(), 0.0f);

    // Fail fast on unknown input keys (matches obs_builder.py).
    for (const auto& s : cfg.policy_inputs) {
        if (!isKnownKey(s.key))
            throw std::runtime_error("ObsBuilder: unknown ONNX input key '" + s.key + "'");
    }
}

void ObsBuilder::resetHistory() {
    std::fill(history_buf_.begin(), history_buf_.end(), 0.0f);
}

void ObsBuilder::resetHistory(int character) {
    const int stride = histStride();
    auto begin = history_buf_.begin() + (size_t)character * stride;
    std::fill(begin, begin + stride, 0.0f);
}

void ObsBuilder::pushAction(int character, const std::vector<float>& actions_raw) {
    if (history_steps_ == 0) return;
    float* base = history_buf_.data() + (size_t)character * histStride();
    if (history_steps_ > 1) {
        // Shift right: slot 0 = newest. Move [0..H-2] -> [1..H-1].
        for (int h = history_steps_ - 1; h >= 1; --h) {
            std::copy(base + (size_t)(h - 1) * num_dofs_, base + (size_t)h * num_dofs_,
                      base + (size_t)h * num_dofs_);
        }
    }
    std::copy(actions_raw.begin(), actions_raw.begin() + num_dofs_, base);
}

Feed ObsBuilder::build(const std::vector<CharacterObsInputs>& chars) {
    if ((int)chars.size() != num_characters_)
        throw std::runtime_error("ObsBuilder: character count mismatch");

    const int N = num_characters_;
    const int B = num_bodies_;
    const int D = num_dofs_;
    const int F = cfg_.future_steps();
    const int H = history_steps_ > 0 ? history_steps_ : 1;
    const int M = cfg_.num_collision_primitives();
    const int C = cfg_.num_contact_bodies();
    const int S = cfg_.num_stamina_bodies();

    // Per-key stacked buffers (leading dim N). Only keys the model needs are filled.
    std::unordered_map<std::string, std::vector<float>> buf;
    std::unordered_map<std::string, std::vector<int64_t>> shp;
    std::unordered_map<std::string, bool> boolkey;

    auto reserve = [&](const char* key, std::vector<int64_t> per_char_shape, bool is_bool) {
        if (!cfg_.has_input_key(key)) return;
        int64_t per = 1;
        for (int64_t d : per_char_shape) per *= d;
        buf[key].reserve((size_t)N * per);
        std::vector<int64_t> s;
        s.push_back(N);
        for (int64_t d : per_char_shape) s.push_back(d);
        shp[key] = std::move(s);
        boolkey[key] = is_bool;
    };

    reserve(KEY_DOF_POS, {D}, false);
    reserve(KEY_DOF_VEL, {D}, false);
    reserve(KEY_FUT_DOF_POS, {F, D}, false);
    reserve(KEY_FUT_DOF_VEL, {F, D}, false);
    reserve(KEY_POS, {B, 3}, false);
    reserve(KEY_ROT, {B, 4}, false);
    reserve(KEY_VEL, {B, 3}, false);
    reserve(KEY_ANG, {B, 3}, false);
    reserve(KEY_GROUND, {}, false);
    reserve(KEY_HIST, {H, D}, false);
    reserve(KEY_FUT_POS, {F, B, 3}, false);
    reserve(KEY_FUT_ROT, {F, B, 4}, false);
    reserve(KEY_FUT_VEL, {F, B, 3}, false);
    reserve(KEY_FUT_ANG, {F, B, 3}, false);
    reserve(KEY_CONTACTS, {C}, true);
    reserve(KEY_STAMINA, {S}, false);
    reserve(KEY_CP_POS, {M, 3}, false);
    reserve(KEY_CP_ROT, {M, 4}, false);
    reserve(KEY_CP_VEL, {M, 3}, false);
    reserve(KEY_CP_RADIUS, {M}, false);
    reserve(KEY_CP_EXTENT, {M}, false);
    reserve(KEY_CP_DAMAGE, {M}, false);
    reserve(KEY_CP_MASS, {M}, false);
    reserve(KEY_CP_SHAPE, {M, 2}, false);
    reserve(KEY_CP_VALID, {M}, false);

    auto append = [&](const char* key, const float* data, size_t n) {
        auto it = buf.find(key);
        if (it == buf.end()) return;  // model does not request this key
        it->second.insert(it->second.end(), data, data + n);
    };
    auto append_vec3 = [&](const char* key, const std::vector<std::array<float, 3>>& v) {
        if (buf.find(key) == buf.end()) return;
        for (const auto& e : v) append(key, e.data(), 3);
    };
    auto append_vec4 = [&](const char* key, const std::vector<std::array<float, 4>>& v) {
        if (buf.find(key) == buf.end()) return;
        for (const auto& e : v) append(key, e.data(), 4);
    };

    for (int c = 0; c < N; ++c) {
        const CharacterObsInputs& in = chars[c];
        if (!in.state || !in.future)
            throw std::runtime_error("ObsBuilder: missing per-character state/future");
        const ArticulationState& st = *in.state;
        const MotionFutureData& fut = *in.future;

        append(KEY_DOF_POS, st.dof_pos.data(), st.dof_pos.size());
        append(KEY_DOF_VEL, st.dof_vel.data(), st.dof_vel.size());
        append(KEY_FUT_DOF_POS, fut.dof_pos.data(), fut.dof_pos.size());
        append(KEY_FUT_DOF_VEL, fut.dof_vel.data(), fut.dof_vel.size());
        append_vec3(KEY_POS, st.body_pos);
        append_vec4(KEY_ROT, st.body_rot);
        append_vec3(KEY_VEL, st.body_lin_vel);
        append_vec3(KEY_ANG, st.body_ang_vel);
        { float g = in.ground_height; append(KEY_GROUND, &g, 1); }
        append(KEY_HIST, history_buf_.data() + (size_t)c * histStride(), (size_t)H * D);
        append(KEY_FUT_POS, fut.pos.data(), fut.pos.size());
        append(KEY_FUT_ROT, fut.rot.data(), fut.rot.size());
        append(KEY_FUT_VEL, fut.vel.data(), fut.vel.size());
        append(KEY_FUT_ANG, fut.ang_vel.data(), fut.ang_vel.size());

        if (buf.find(KEY_CONTACTS) != buf.end()) {
            if (!in.body_contacts || (int)in.body_contacts->size() != C)
                throw std::runtime_error("ObsBuilder: body_contacts size mismatch");
            for (int i = 0; i < C; ++i) {
                float v = (*in.body_contacts)[i] ? 1.0f : 0.0f;
                append(KEY_CONTACTS, &v, 1);
            }
        }
        if (buf.find(KEY_STAMINA) != buf.end()) {
            if (!in.body_stamina || (int)in.body_stamina->size() != S)
                throw std::runtime_error("ObsBuilder: body_stamina size mismatch");
            append(KEY_STAMINA, in.body_stamina->data(), S);
        }
        if (buf.find(KEY_CP_POS) != buf.end()) {
            if (!in.primitives || in.primitives->M != M)
                throw std::runtime_error("ObsBuilder: collision primitives size mismatch");
            const CollisionPrimitives& cp = *in.primitives;
            append(KEY_CP_POS, cp.pos.data(), cp.pos.size());
            append(KEY_CP_ROT, cp.rot.data(), cp.rot.size());
            append(KEY_CP_VEL, cp.lin_vel.data(), cp.lin_vel.size());
            append(KEY_CP_RADIUS, cp.radius.data(), cp.radius.size());
            append(KEY_CP_EXTENT, cp.extent_z.data(), cp.extent_z.size());
            append(KEY_CP_DAMAGE, cp.damage.data(), cp.damage.size());
            append(KEY_CP_MASS, cp.mass.data(), cp.mass.size());
            append(KEY_CP_SHAPE, cp.shape.data(), cp.shape.size());
            append(KEY_CP_VALID, cp.valid.data(), cp.valid.size());
        }
    }

    // Emit in the exact order the sidecar declares its inputs.
    Feed feed;
    for (const auto& spec : cfg_.policy_inputs) {
        auto it = buf.find(spec.key);
        if (it == buf.end())
            throw std::runtime_error("ObsBuilder: no data for key '" + spec.key + "'");
        Tensor t;
        t.data = std::move(it->second);
        t.shape = shp[spec.key];
        t.is_bool = boolkey[spec.key];

        if (!spec.shape.empty()) {
            // Validate per-character element count against the sidecar (batch dim
            // aside): product of declared shape[1:] * N == data size.
            int64_t per = 1;
            for (size_t d = 1; d < spec.shape.size(); ++d) per *= spec.shape[d];
            if ((int64_t)t.data.size() != per * N)
                throw std::runtime_error("ObsBuilder: size mismatch for '" + spec.name + "'");
        }
        feed.names.push_back(spec.name);
        feed.tensors.push_back(std::move(t));
    }
    return feed;
}

}  // namespace ovpp
