#include "Renderer.h"

#include <glad/gl.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ovpp {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// ---- shaders --------------------------------------------------------------
const char* kVertSrc = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
out vec3 vNormalW;
void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vNormalW = mat3(transpose(inverse(uModel))) * aNormal;
    gl_Position = uProj * uView * world;
}
)";

const char* kFragSrc = R"(#version 330 core
in vec3 vNormalW;
uniform vec3 uColor;
uniform float uAlpha;
uniform int uUnlit;
uniform vec3 uLightDir;
out vec4 FragColor;
void main() {
    if (uUnlit == 1) {
        FragColor = vec4(uColor, uAlpha);
        return;
    }
    vec3 N = normalize(vNormalW);
    vec3 L = normalize(-uLightDir);
    float diff = max(dot(N, L), 0.0);
    float ambient = 0.35;
    float intensity = ambient + (1.0 - ambient) * diff;
    FragColor = vec4(uColor * intensity, uAlpha);
}
)";

GLuint compileShader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
        throw std::runtime_error(std::string("shader compile failed: ") + log);
    }
    return sh;
}

GLuint linkProgram() {
    GLuint vs = compileShader(GL_VERTEX_SHADER, kVertSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFragSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        throw std::runtime_error(std::string("program link failed: ") + log);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// Shortest-arc rotation that maps ``from`` onto ``to`` (mirrors the helper in
// SmplArticulation so the rendered capsule matches the PhysX shape exactly).
glm::quat quatFromTo(const glm::vec3& from, const glm::vec3& to) {
    glm::vec3 a = glm::normalize(from);
    glm::vec3 b = glm::normalize(to);
    float d = glm::dot(a, b);
    if (d >= 1.0f - 1e-6f) return glm::quat(1, 0, 0, 0);
    if (d <= -1.0f + 1e-6f) {
        glm::vec3 axis = glm::cross(glm::vec3(1, 0, 0), a);
        if (glm::dot(axis, axis) < 1e-6f) axis = glm::cross(glm::vec3(0, 1, 0), a);
        axis = glm::normalize(axis);
        return glm::angleAxis(kPi, axis);
    }
    glm::vec3 c = glm::cross(a, b);
    glm::quat q(1.0f + d, c.x, c.y, c.z);  // glm::quat(w, x, y, z)
    return glm::normalize(q);
}

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
};

struct GpuMesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei indexCount = 0;
    GLenum mode = GL_TRIANGLES;

    void upload(const std::vector<Vertex>& verts,
                const std::vector<unsigned int>& idx, GLenum drawMode) {
        mode = drawMode;
        indexCount = static_cast<GLsizei>(idx.size());
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex),
                     verts.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(unsigned int),
                     idx.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                              (void*)offsetof(Vertex, pos));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                              (void*)offsetof(Vertex, normal));
        glBindVertexArray(0);
    }

    void draw() const {
        glBindVertexArray(vao);
        glDrawElements(mode, indexCount, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
    }

    void destroy() {
        if (ebo) glDeleteBuffers(1, &ebo);
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);
        vao = vbo = ebo = 0;
    }
};

// Unit cube spanning [-1, 1] on each axis (half-extent 1) so a scale by the
// box half-extents reproduces the PhysX PxBoxGeometry exactly.
GpuMesh makeCube() {
    const glm::vec3 n[6] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                            {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
    // Per-face corner offsets (CCW seen from outside).
    auto face = [](glm::vec3 nrm) {
        std::vector<glm::vec3> c;
        glm::vec3 a = std::abs(nrm.x) > 0.5f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        glm::vec3 u = glm::normalize(glm::cross(nrm, a));
        glm::vec3 v = glm::normalize(glm::cross(nrm, u));
        c.push_back(nrm - u - v);
        c.push_back(nrm + u - v);
        c.push_back(nrm + u + v);
        c.push_back(nrm - u + v);
        return c;
    };
    std::vector<Vertex> verts;
    std::vector<unsigned int> idx;
    for (int f = 0; f < 6; ++f) {
        auto corners = face(n[f]);
        unsigned int base = static_cast<unsigned int>(verts.size());
        for (auto& c : corners) verts.push_back({c, n[f]});
        idx.insert(idx.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    }
    GpuMesh m;
    m.upload(verts, idx, GL_TRIANGLES);
    return m;
}

// Capsule along local +X (matching the PhysX capsule axis), radius ``r`` and
// cylinder half-length ``h``, built as two hemispheres joined by a cylinder.
GpuMesh makeCapsule(float r, float h, int sectors = 24, int stacks = 10) {
    struct Ring { float x; float radius; float nx; };
    std::vector<Ring> rings;
    for (int i = 0; i <= stacks; ++i) {  // +X hemisphere: pole -> equator
        float u = (kPi * 0.5f) * (1.0f - (float)i / stacks);
        rings.push_back({h + r * std::sin(u), r * std::cos(u), std::sin(u)});
    }
    for (int i = 0; i <= stacks; ++i) {  // -X hemisphere: equator -> pole
        float u = (kPi * 0.5f) * ((float)i / stacks);
        rings.push_back({-h - r * std::sin(u), r * std::cos(u), -std::sin(u)});
    }

    std::vector<Vertex> verts;
    const int stride = sectors + 1;
    for (const Ring& ring : rings) {
        float nr = ring.radius / r;  // radial component of the surface normal
        for (int j = 0; j <= sectors; ++j) {
            float a = 2.0f * kPi * (float)j / sectors;
            float ca = std::cos(a), sa = std::sin(a);
            glm::vec3 pos(ring.x, ring.radius * ca, ring.radius * sa);
            glm::vec3 nrm = glm::normalize(glm::vec3(ring.nx, nr * ca, nr * sa));
            verts.push_back({pos, nrm});
        }
    }

    std::vector<unsigned int> idx;
    for (size_t k = 0; k + 1 < rings.size(); ++k) {
        for (int j = 0; j < sectors; ++j) {
            unsigned int a = static_cast<unsigned int>(k * stride + j);
            unsigned int b = static_cast<unsigned int>((k + 1) * stride + j);
            idx.insert(idx.end(), {a, b, b + 1, a, b + 1, a + 1});
        }
    }
    GpuMesh m;
    m.upload(verts, idx, GL_TRIANGLES);
    return m;
}

// Z-up ground grid centered at the origin.
GpuMesh makeGrid(int halfCells = 20, float spacing = 0.5f) {
    std::vector<Vertex> verts;
    std::vector<unsigned int> idx;
    float ext = halfCells * spacing;
    glm::vec3 up(0, 0, 1);
    unsigned int n = 0;
    for (int i = -halfCells; i <= halfCells; ++i) {
        float c = i * spacing;
        verts.push_back({{c, -ext, 0}, up});
        verts.push_back({{c, ext, 0}, up});
        verts.push_back({{-ext, c, 0}, up});
        verts.push_back({{ext, c, 0}, up});
        idx.insert(idx.end(), {n, n + 1, n + 2, n + 3});
        n += 4;
    }
    GpuMesh m;
    m.upload(verts, idx, GL_LINES);
    return m;
}

// What and how to draw a single body: a mesh plus the body-local transform
// that places it in the link frame (so the world matrix is just T*R of the
// reported body pose).
struct BodyRender {
    bool valid = false;
    bool ownsMesh = false;   // capsule meshes are per-body; the cube is shared
    const GpuMesh* mesh = nullptr;
    GpuMesh ownMesh;
    glm::mat4 localXf{1.0f};
};

glm::mat4 worldXf(const std::array<float, 3>& pos, const std::array<float, 4>& rot) {
    glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(pos[0], pos[1], pos[2]));
    glm::quat q(rot[3], rot[0], rot[1], rot[2]);  // xyzw -> glm wxyz
    return T * glm::mat4_cast(q);
}

}  // namespace

struct Renderer::Impl {
    GLFWwindow* window = nullptr;
    GLuint prog = 0;
    GLint uModel = -1, uView = -1, uProj = -1;
    GLint uColor = -1, uAlpha = -1, uUnlit = -1, uLightDir = -1;

    GpuMesh cube;
    GpuMesh grid;
    GpuMesh unitSphere;  // radius 1 (drawn scaled by projectile radius)
    std::vector<BodyRender> bodies;

    // Per-projectile meshes, built lazily from the first snapshot.
    std::vector<GpuMesh> projMeshes;
    bool projBuilt = false;

    // Orbit camera (Z-up). Target follows the sim root each frame plus pan.
    float yaw = -0.6f, pitch = 0.35f, dist = 3.5f;
    glm::vec3 panOffset{0.0f};
    glm::vec3 followTarget{0, 0, 1.0f};

    // Input state.
    bool leftDown = false, rightDown = false;
    double lastX = 0, lastY = 0;
    bool isPaused = false;
    bool stepReq = false;
    bool throwReq = false;

    void resetCamera() {
        yaw = -0.6f;
        pitch = 0.35f;
        dist = 3.5f;
        panOffset = glm::vec3(0.0f);
    }

    glm::vec3 cameraEye(const glm::vec3& target) const {
        float cp = std::cos(pitch), sp = std::sin(pitch);
        glm::vec3 dir(cp * std::cos(yaw), cp * std::sin(yaw), sp);
        return target + dir * dist;
    }
};

namespace {

Renderer::Impl* implOf(GLFWwindow* w) {
    return static_cast<Renderer::Impl*>(glfwGetWindowUserPointer(w));
}

void keyCb(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    Renderer::Impl* s = implOf(w);
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, GLFW_TRUE);
    else if (key == GLFW_KEY_SPACE) s->isPaused = !s->isPaused;
    else if (key == GLFW_KEY_RIGHT) s->stepReq = true;
    else if (key == GLFW_KEY_R) s->resetCamera();
    else if (key == GLFW_KEY_J) s->throwReq = true;
}

void mouseBtnCb(GLFWwindow* w, int button, int action, int) {
    Renderer::Impl* s = implOf(w);
    bool down = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_LEFT) s->leftDown = down;
    else if (button == GLFW_MOUSE_BUTTON_RIGHT) s->rightDown = down;
    glfwGetCursorPos(w, &s->lastX, &s->lastY);
}

void cursorCb(GLFWwindow* w, double x, double y) {
    Renderer::Impl* s = implOf(w);
    double dx = x - s->lastX, dy = y - s->lastY;
    s->lastX = x;
    s->lastY = y;
    if (s->leftDown) {
        s->yaw -= (float)dx * 0.01f;
        s->pitch += (float)dy * 0.01f;
        const float lim = kPi * 0.5f - 0.05f;
        if (s->pitch > lim) s->pitch = lim;
        if (s->pitch < -lim) s->pitch = -lim;
    } else if (s->rightDown) {
        // Pan in the camera's right/up plane.
        glm::vec3 target = s->followTarget + s->panOffset;
        glm::vec3 eye = s->cameraEye(target);
        glm::vec3 fwd = glm::normalize(target - eye);
        glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0, 0, 1)));
        glm::vec3 up = glm::normalize(glm::cross(right, fwd));
        float scale = s->dist * 0.0015f;
        s->panOffset += (-right * (float)dx + up * (float)dy) * scale;
    }
}

void scrollCb(GLFWwindow* w, double, double yoff) {
    Renderer::Impl* s = implOf(w);
    s->dist *= std::pow(0.9f, (float)yoff);
    if (s->dist < 0.3f) s->dist = 0.3f;
    if (s->dist > 50.0f) s->dist = 50.0f;
}

}  // namespace

Renderer::Renderer(int width, int height, const MjcfModel& mjcf)
    : p_(new Impl()) {
    Impl& s = *p_;

    if (!glfwInit()) throw std::runtime_error("glfwInit failed");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    s.window = glfwCreateWindow(width, height, "OVMiniBrain - Hierarchical SMPL", nullptr, nullptr);
    if (!s.window) {
        glfwTerminate();
        throw std::runtime_error("glfwCreateWindow failed");
    }
    glfwMakeContextCurrent(s.window);
    glfwSwapInterval(1);

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        glfwDestroyWindow(s.window);
        glfwTerminate();
        throw std::runtime_error("gladLoadGL failed");
    }

    glfwSetWindowUserPointer(s.window, &s);
    glfwSetKeyCallback(s.window, keyCb);
    glfwSetMouseButtonCallback(s.window, mouseBtnCb);
    glfwSetCursorPosCallback(s.window, cursorCb);
    glfwSetScrollCallback(s.window, scrollCb);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s.prog = linkProgram();
    s.uModel = glGetUniformLocation(s.prog, "uModel");
    s.uView = glGetUniformLocation(s.prog, "uView");
    s.uProj = glGetUniformLocation(s.prog, "uProj");
    s.uColor = glGetUniformLocation(s.prog, "uColor");
    s.uAlpha = glGetUniformLocation(s.prog, "uAlpha");
    s.uUnlit = glGetUniformLocation(s.prog, "uUnlit");
    s.uLightDir = glGetUniformLocation(s.prog, "uLightDir");

    s.cube = makeCube();
    s.grid = makeGrid();
    s.unitSphere = makeCapsule(1.0f, 0.0f);  // capsule with zero cylinder == sphere

    // Pre-build the per-body draw data from the MJCF geom list.
    s.bodies.resize(mjcf.bodies.size());
    for (size_t b = 0; b < mjcf.bodies.size(); ++b) {
        const Geom& g = mjcf.bodies[b].geom;
        BodyRender& br = s.bodies[b];
        if (g.type == GeomType::Box) {
            if (g.size[0] <= 0 && g.size[1] <= 0 && g.size[2] <= 0) continue;
            glm::vec3 pos(g.pos[0], g.pos[1], g.pos[2]);
            glm::quat q((float)g.quat[0], (float)g.quat[1], (float)g.quat[2],
                        (float)g.quat[3]);  // wxyz
            glm::mat4 M = glm::translate(glm::mat4(1.0f), pos);
            M *= glm::mat4_cast(glm::normalize(q));
            M = glm::scale(M, glm::vec3((float)g.size[0], (float)g.size[1],
                                        (float)g.size[2]));
            br.localXf = M;
            br.mesh = &s.cube;
            br.ownsMesh = false;
            br.valid = true;
        } else {
            glm::vec3 a(g.fromto_a[0], g.fromto_a[1], g.fromto_a[2]);
            glm::vec3 bb(g.fromto_b[0], g.fromto_b[1], g.fromto_b[2]);
            glm::vec3 seg = bb - a;
            float len = glm::length(seg);
            if (g.radius <= 0 || len <= 1e-9f) continue;
            float halfH = 0.5f * len;
            glm::vec3 center = 0.5f * (a + bb);
            glm::quat q = quatFromTo(glm::vec3(1, 0, 0), seg);
            glm::mat4 M = glm::translate(glm::mat4(1.0f), center);
            M *= glm::mat4_cast(q);
            br.localXf = M;
            br.ownMesh = makeCapsule((float)g.radius, halfH);
            br.mesh = &br.ownMesh;
            br.ownsMesh = true;
            br.valid = true;
        }
    }
}

Renderer::~Renderer() {
    if (!p_) return;
    Impl& s = *p_;
    if (s.window) {
        glfwMakeContextCurrent(s.window);
        for (auto& br : s.bodies)
            if (br.ownsMesh) br.ownMesh.destroy();
        for (auto& m : s.projMeshes) m.destroy();
        s.cube.destroy();
        s.grid.destroy();
        s.unitSphere.destroy();
        if (s.prog) glDeleteProgram(s.prog);
        glfwDestroyWindow(s.window);
    }
    glfwTerminate();
}

void Renderer::drawFrame(const std::vector<const ArticulationState*>& sims,
                         const std::vector<std::vector<std::array<float, 3>>>& refPos,
                         const std::vector<std::vector<std::array<float, 4>>>& refRot,
                         const ProjectileSnapshot* proj) {
    Impl& s = *p_;
    glfwPollEvents();

    // Camera follows character 0.
    const ArticulationState* c0 = sims.empty() ? nullptr : sims[0];
    if (c0) s.followTarget = glm::vec3(c0->root_pos[0], c0->root_pos[1], c0->root_pos[2]);
    glm::vec3 target = s.followTarget + s.panOffset;
    glm::vec3 eye = s.cameraEye(target);

    int fbw, fbh;
    glfwGetFramebufferSize(s.window, &fbw, &fbh);
    if (fbh == 0) fbh = 1;
    glViewport(0, 0, fbw, fbh);

    glClearColor(0.12f, 0.13f, 0.16f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 projMat = glm::perspective(glm::radians(45.0f),
                                         (float)fbw / (float)fbh, 0.05f, 200.0f);
    glm::mat4 view = glm::lookAt(eye, target, glm::vec3(0, 0, 1));

    glUseProgram(s.prog);
    glUniformMatrix4fv(s.uView, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(s.uProj, 1, GL_FALSE, glm::value_ptr(projMat));
    glm::vec3 lightDir = glm::normalize(glm::vec3(-0.4f, -0.55f, -0.75f));
    glUniform3fv(s.uLightDir, 1, glm::value_ptr(lightDir));

    // Ground grid (unlit lines).
    glm::mat4 I(1.0f);
    glUniformMatrix4fv(s.uModel, 1, GL_FALSE, glm::value_ptr(I));
    glUniform1i(s.uUnlit, 1);
    glUniform1f(s.uAlpha, 1.0f);
    glUniform3f(s.uColor, 0.30f, 0.32f, 0.36f);
    s.grid.draw();

    // Per-character solid bodies with a small color palette.
    static const glm::vec3 kPalette[] = {
        {0.30f, 0.62f, 0.90f}, {0.90f, 0.45f, 0.40f}, {0.45f, 0.80f, 0.45f},
        {0.80f, 0.70f, 0.35f}, {0.65f, 0.45f, 0.85f}, {0.40f, 0.80f, 0.80f},
    };
    const int B = (int)s.bodies.size();
    glUniform1i(s.uUnlit, 0);
    glUniform1f(s.uAlpha, 1.0f);
    for (int c = 0; c < (int)sims.size(); ++c) {
        const ArticulationState& sim = *sims[c];
        const glm::vec3& col = kPalette[c % (int)(sizeof(kPalette) / sizeof(kPalette[0]))];
        glUniform3f(s.uColor, col.r, col.g, col.b);
        for (int b = 0; b < B && b < (int)sim.body_pos.size(); ++b) {
            if (!s.bodies[b].valid) continue;
            glm::mat4 model =
                worldXf(sim.body_pos[b], sim.body_rot[b]) * s.bodies[b].localXf;
            glUniformMatrix4fv(s.uModel, 1, GL_FALSE, glm::value_ptr(model));
            s.bodies[b].mesh->draw();
        }
    }

    // Projectiles (opaque). Build per-slot meshes lazily from the snapshot.
    if (proj && proj->P > 0) {
        if (!s.projBuilt) {
            s.projMeshes.resize(proj->P);
            for (int i = 0; i < proj->P; ++i) {
                bool isBox = proj->shape[i][0] > 0.5f;
                bool isSphere = proj->shape[i][1] > 0.5f;
                if (isBox) {
                    s.projMeshes[i] = makeCube();
                } else if (isSphere) {
                    s.projMeshes[i] = makeCapsule(proj->radius[i], 0.0f);
                } else {  // capsule (cylinder length == extent_z)
                    s.projMeshes[i] = makeCapsule(proj->radius[i], 0.5f * proj->extent_z[i]);
                }
            }
            s.projBuilt = true;
        }
        glUniform3f(s.uColor, 0.92f, 0.86f, 0.30f);
        for (int i = 0; i < proj->P; ++i) {
            if (proj->active[i] < 0.5f) continue;
            glm::mat4 model = worldXf(proj->pos[i], proj->rot[i]);
            if (proj->shape[i][0] > 0.5f) {  // box: cube scaled by half-extent
                float h = 0.5f * proj->extent_z[i];
                model = glm::scale(model, glm::vec3(h, h, h));
            }
            glUniformMatrix4fv(s.uModel, 1, GL_FALSE, glm::value_ptr(model));
            s.projMeshes[i].draw();
        }
    }

    // Reference ghosts: translucent, depth-write off.
    glDepthMask(GL_FALSE);
    glUniform1f(s.uAlpha, 0.35f);
    glUniform3f(s.uColor, 0.95f, 0.55f, 0.20f);
    for (int c = 0; c < (int)refPos.size() && c < (int)refRot.size(); ++c) {
        const auto& rp = refPos[c];
        const auto& rr = refRot[c];
        for (int b = 0; b < B && b < (int)rp.size() && b < (int)rr.size(); ++b) {
            if (!s.bodies[b].valid) continue;
            glm::mat4 model = worldXf(rp[b], rr[b]) * s.bodies[b].localXf;
            glUniformMatrix4fv(s.uModel, 1, GL_FALSE, glm::value_ptr(model));
            s.bodies[b].mesh->draw();
        }
    }
    glDepthMask(GL_TRUE);

    glfwSwapBuffers(s.window);
}

bool Renderer::shouldClose() const {
    return glfwWindowShouldClose(p_->window) != 0;
}

bool Renderer::paused() const { return p_->isPaused; }

bool Renderer::stepRequested() {
    bool r = p_->stepReq;
    p_->stepReq = false;
    return r;
}

bool Renderer::throwRequested() {
    bool r = p_->throwReq;
    p_->throwReq = false;
    return r;
}

}  // namespace ovpp
