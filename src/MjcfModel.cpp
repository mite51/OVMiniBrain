#include "MjcfModel.h"

#include <pugixml.hpp>

#include <sstream>
#include <stdexcept>

namespace ovpp {

namespace {

std::vector<double> parse_doubles(const char* s) {
    std::vector<double> out;
    if (!s) return out;
    std::istringstream iss(s);
    double v;
    while (iss >> v) out.push_back(v);
    return out;
}

void parse_geom(const pugi::xml_node& g, Geom& geom) {
    std::string type = g.attribute("type").as_string("box");
    geom.density = g.attribute("density").as_double(1000.0);
    if (type == "capsule") {
        geom.type = GeomType::Capsule;
        auto ft = parse_doubles(g.attribute("fromto").as_string());
        if (ft.size() == 6) {
            geom.fromto_a = {ft[0], ft[1], ft[2]};
            geom.fromto_b = {ft[3], ft[4], ft[5]};
        }
        // MJCF capsule "size" first value is the radius.
        auto sz = parse_doubles(g.attribute("size").as_string());
        if (!sz.empty()) geom.radius = sz[0];
    } else {
        geom.type = GeomType::Box;
        auto pos = parse_doubles(g.attribute("pos").as_string("0 0 0"));
        if (pos.size() == 3) geom.pos = {pos[0], pos[1], pos[2]};
        auto q = parse_doubles(g.attribute("quat").as_string("1 0 0 0"));
        if (q.size() == 4) geom.quat = {q[0], q[1], q[2], q[3]};
        auto sz = parse_doubles(g.attribute("size").as_string("0 0 0"));
        if (sz.size() == 3) geom.size = {sz[0], sz[1], sz[2]};
    }
}

void walk(const pugi::xml_node& body_node, int parent, MjcfModel& model) {
    MjcfBody b;
    b.name = body_node.attribute("name").as_string();
    b.parent = parent;
    b.is_root = (parent < 0);
    auto pos = parse_doubles(body_node.attribute("pos").as_string("0 0 0"));
    if (pos.size() == 3) b.pos = {pos[0], pos[1], pos[2]};

    // First (collision) geom child.
    pugi::xml_node g = body_node.child("geom");
    if (g) parse_geom(g, b.geom);

    // Armature from the first hinge joint (uniform across axes for SMPL).
    for (pugi::xml_node j = body_node.child("joint"); j; j = j.next_sibling("joint")) {
        std::string jtype = j.attribute("type").as_string("hinge");
        if (jtype == "hinge") {
            b.armature = j.attribute("armature").as_double(0.0);
            break;
        }
    }

    int my_index = static_cast<int>(model.bodies.size());
    model.bodies.push_back(b);

    for (pugi::xml_node child = body_node.child("body"); child;
         child = child.next_sibling("body")) {
        walk(child, my_index, model);
    }
}

}  // namespace

MjcfModel MjcfModel::load(const std::string& path) {
    pugi::xml_document doc;
    pugi::xml_parse_result res = doc.load_file(path.c_str());
    if (!res)
        throw std::runtime_error("Failed to parse MJCF '" + path + "': " +
                                 res.description());

    pugi::xml_node worldbody = doc.child("mujoco").child("worldbody");
    pugi::xml_node root_body = worldbody.child("body");
    if (!root_body)
        throw std::runtime_error("MJCF has no worldbody/body: " + path);

    MjcfModel model;
    walk(root_body, -1, model);
    return model;
}

}  // namespace ovpp
