#include "ObsBuilder.h"
#include <iostream>
#include <stdexcept>

using namespace ovpp;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() try {
    Config cfg;
    cfg.body_names.resize(24);
    cfg.dof_names.resize(69);
    cfg.future_step_indices = {1, 3};
    cfg.policy_inputs = {
        {"q", "current.dof_pos", {1, 69}},
        {"dq", "current.dof_vel", {1, 69}},
        {"qref", "mimic.future_dof_pos", {1, 2, 69}},
        {"dqref", "mimic.future_dof_vel", {1, 2, 69}},
        {"stamina", "body_stamina", {1, 23}},
        {"history", "historical.actions", {1, 1, 69}},
    };
    ObsBuilder builder(cfg, 2);
    std::vector<ArticulationState> states(2);
    std::vector<MotionFutureData> future(2);
    std::vector<std::vector<float>> stamina(2, std::vector<float>(23));
    std::vector<CharacterObsInputs> inputs(2);
    for (int c=0; c<2; ++c) {
        states[c].dof_pos.resize(69);
        states[c].dof_vel.resize(69);
        future[c].dof_pos.resize(138);
        future[c].dof_vel.resize(138);
        for (int i=0; i<69; ++i) {
            states[c].dof_pos[i] = c*1000+i;
            states[c].dof_vel[i] = c*2000+i;
        }
        for (int i=0; i<138; ++i) {
            future[c].dof_pos[i] = c*3000+i;
            future[c].dof_vel[i] = c*4000+i;
        }
        std::fill(stamina[c].begin(), stamina[c].end(), 0.2f+c*0.8f);
        inputs[c].state = &states[c];
        inputs[c].future = &future[c];
        inputs[c].body_stamina = &stamina[c];
        builder.pushAction(c, std::vector<float>(69, 5.0f+c));
    }
    Feed feed = builder.build(inputs);
    for (int c=0; c<2; ++c) {
        for (int i=0; i<69; ++i) {
            require(feed.tensors[0].data[c*69+i] == c*1000+i, "current pose ordering");
            require(feed.tensors[1].data[c*69+i] == c*2000+i, "current velocity ordering");
            require(feed.tensors[5].data[c*69+i] == 5.0f+c, "action history ordering");
        }
        for (int i=0; i<138; ++i) {
            require(feed.tensors[2].data[c*138+i] == c*3000+i, "future pose ordering");
            require(feed.tensors[3].data[c*138+i] == c*4000+i, "future velocity ordering");
        }
        require(feed.tensors[4].data[c*23] == 0.2f+c*0.8f, "stamina ordering");
    }
    require(feed.tensors[2].shape == std::vector<int64_t>({2,2,69}), "future batch shape");
    builder.resetHistory(1);
    feed = builder.build(inputs);
    require(feed.tensors[5].data[0] == 5.0f && feed.tensors[5].data[69] == 0, "per-character reset");
    future[0].dof_pos.pop_back();
    bool rejected=false;
    try { builder.build(inputs); } catch(const std::runtime_error&) { rejected=true; }
    require(rejected, "malformed future must be rejected");
    std::cout << "Mini-brain joint state, future, stamina and history mapping passed\n";
    return 0;
} catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
