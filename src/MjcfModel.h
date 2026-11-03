// Minimal MJCF reader for the SMPL humanoid.
//
// Parses smpl_humanoid.xml into a flat, DFS-ordered body list (which matches
// the YAML ``body_names`` order). Each non-root body carries its offset from
// the parent (the joint anchor), one collision geom (box or capsule) with its
// density, and the joint armature. PhysX-specific construction lives in
// SmplArticulation; this struct is pure data.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace ovpp {

enum class GeomType { Box, Capsule };

struct Geom {
    GeomType type = GeomType::Box;
    double density = 1000.0;
    std::array<double, 3> pos{{0, 0, 0}};      // box: center; capsule: unused
    std::array<double, 4> quat{{1, 0, 0, 0}};  // wxyz; box only
    std::array<double, 3> size{{0, 0, 0}};     // box half-extents
    std::array<double, 3> fromto_a{{0, 0, 0}};  // capsule endpoint A
    std::array<double, 3> fromto_b{{0, 0, 0}};  // capsule endpoint B
    double radius = 0.0;                        // capsule radius
};

struct MjcfBody {
    std::string name;
    int parent = -1;                       // index into bodies; -1 for root
    std::array<double, 3> pos{{0, 0, 0}};  // offset from parent body frame
    Geom geom;
    bool is_root = false;
    double armature = 0.0;                 // per-axis joint armature (0 for root)
};

class MjcfModel {
public:
    std::vector<MjcfBody> bodies;  // DFS preorder (== YAML body_names order)

    // Parse the MJCF at ``path``. Throws std::runtime_error on failure.
    static MjcfModel load(const std::string& path);
};

}  // namespace ovpp
