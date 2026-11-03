// Optional real-time OpenGL viewer for the SMPL tracking harness.
//
// Draws the simulated PhysX articulation solid and overlays the motion-cache
// reference pose as a translucent ghost. GL 3.3 core via GLFW + glad + glm.
// Entirely self-contained and gated behind the --render flag in main.cpp; the
// headless parity path never touches this module.
#pragma once

#include "CollisionPrimitives.h"
#include "MjcfModel.h"
#include "SmplArticulation.h"

#include <array>
#include <memory>
#include <vector>

namespace ovpp {

class Renderer {
public:
    // Creates a GL 3.3 core window of the given size and pre-builds one mesh
    // per body from the MJCF geom list. Throws std::runtime_error on failure.
    Renderer(int width, int height, const MjcfModel& mjcf);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Render one frame for N characters. ``sims`` are the simulated states;
    // ``refPos``/``refRot`` are the per-character reference body world poses for
    // the current tick ([N][B], sim coords, xyzw quats) drawn as translucent
    // ghosts. ``proj`` (optional) draws active thrown projectiles.
    void drawFrame(const std::vector<const ArticulationState*>& sims,
                   const std::vector<std::vector<std::array<float, 3>>>& refPos,
                   const std::vector<std::vector<std::array<float, 4>>>& refRot,
                   const ProjectileSnapshot* proj = nullptr);

    bool shouldClose() const;
    bool paused() const;
    bool stepRequested();   // returns true once per single-step request
    bool throwRequested();  // returns true once per 'J' key press

    struct Impl;  // public so the GLFW C callbacks can reach the state

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace ovpp
