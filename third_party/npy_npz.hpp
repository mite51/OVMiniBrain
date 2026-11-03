// Minimal, dependency-free NumPy .npy / .npz writer.
//
// Produces real .npz files (a ZIP archive of .npy members, STORED / no
// compression) so the existing scripts/parity_compare.py can np.load() the
// output unchanged. Supports the three dtypes the parity dump needs:
//
//   - float32          ('<f4')   N-dimensional arrays
//   - int64 scalar     ('<i8')   meta_step_count
//   - unicode scalar   ('<U..')  meta_motion_file / meta_simulator strings
//
// Unicode is written as 0-d UCS4 (UTF-32LE) so np.load reads it WITHOUT
// allow_pickle, and .item() returns a Python str.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <stdexcept>

namespace npz {

// ---------------------------------------------------------------------------
// CRC32 (zlib polynomial) for the ZIP central directory.
// ---------------------------------------------------------------------------
inline uint32_t crc32_update(uint32_t crc, const uint8_t* buf, size_t len) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    crc ^= 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i)
        crc = table[(crc ^ buf[i]) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

// A single .npy member built in memory.
struct NpyMember {
    std::string name;        // e.g. "in_current_rigid_body_pos.npy"
    std::vector<uint8_t> bytes;
    uint32_t crc = 0;
};

inline std::string make_npy_header(const std::string& descr,
                                   const std::vector<size_t>& shape) {
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        if (shape.size() == 1 || i + 1 < shape.size())
            shape_str += ",";
        if (i + 1 < shape.size())
            shape_str += " ";
    }
    shape_str += ")";

    std::string dict = "{'descr': '" + descr +
                       "', 'fortran_order': False, 'shape': " + shape_str + ", }";
    // npy v1.0: 6 magic + 2 version + 2 header-len + header (padded to 64).
    size_t prefix = 10;
    size_t total = prefix + dict.size() + 1;  // +1 for trailing '\n'
    size_t pad = (64 - (total % 64)) % 64;
    dict.append(pad, ' ');
    dict.push_back('\n');

    std::string out;
    out.push_back('\x93');
    out += "NUMPY";
    out.push_back('\x01');  // major
    out.push_back('\x00');  // minor
    uint16_t hlen = static_cast<uint16_t>(dict.size());
    out.push_back(static_cast<char>(hlen & 0xFF));
    out.push_back(static_cast<char>((hlen >> 8) & 0xFF));
    out += dict;
    return out;
}

inline NpyMember npy_float32(const std::string& key,
                             const std::vector<size_t>& shape,
                             const float* data, size_t count) {
    NpyMember m;
    m.name = key + ".npy";
    std::string hdr = make_npy_header("<f4", shape);
    m.bytes.insert(m.bytes.end(), hdr.begin(), hdr.end());
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(data);
    m.bytes.insert(m.bytes.end(), raw, raw + count * sizeof(float));
    m.crc = crc32_update(0, m.bytes.data(), m.bytes.size());
    return m;
}

inline NpyMember npy_int64_scalar(const std::string& key, int64_t value) {
    NpyMember m;
    m.name = key + ".npy";
    std::string hdr = make_npy_header("<i8", {});  // 0-d
    m.bytes.insert(m.bytes.end(), hdr.begin(), hdr.end());
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(&value);
    m.bytes.insert(m.bytes.end(), raw, raw + sizeof(int64_t));
    m.crc = crc32_update(0, m.bytes.data(), m.bytes.size());
    return m;
}

inline NpyMember npy_unicode_scalar(const std::string& key,
                                    const std::string& utf8) {
    // 0-d '<U{n}' array: UCS4 little-endian codepoints (ASCII only here).
    NpyMember m;
    m.name = key + ".npy";
    size_t n = utf8.size();
    std::string descr = "<U" + std::to_string(n);
    std::string hdr = make_npy_header(descr, {});
    m.bytes.insert(m.bytes.end(), hdr.begin(), hdr.end());
    for (char c : utf8) {
        uint32_t cp = static_cast<uint8_t>(c);
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(&cp);
        m.bytes.insert(m.bytes.end(), raw, raw + 4);
    }
    m.crc = crc32_update(0, m.bytes.data(), m.bytes.size());
    return m;
}

// ---------------------------------------------------------------------------
// Write the members as a STORED (uncompressed) ZIP -> .npz.
// ---------------------------------------------------------------------------
inline void put16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(x & 0xFF); v.push_back((x >> 8) & 0xFF);
}
inline void put32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(x & 0xFF); v.push_back((x >> 8) & 0xFF);
    v.push_back((x >> 16) & 0xFF); v.push_back((x >> 24) & 0xFF);
}

inline void write_npz(const std::string& path, const std::vector<NpyMember>& members) {
    std::vector<uint8_t> out;
    struct CD { std::string name; uint32_t crc, size, offset; };
    std::vector<CD> central;

    for (const auto& m : members) {
        uint32_t offset = static_cast<uint32_t>(out.size());
        uint32_t size = static_cast<uint32_t>(m.bytes.size());
        // Local file header.
        put32(out, 0x04034b50);
        put16(out, 20);            // version needed
        put16(out, 0);             // flags
        put16(out, 0);             // method: stored
        put16(out, 0); put16(out, 0);  // mod time/date
        put32(out, m.crc);
        put32(out, size);          // compressed
        put32(out, size);          // uncompressed
        put16(out, static_cast<uint16_t>(m.name.size()));
        put16(out, 0);             // extra len
        out.insert(out.end(), m.name.begin(), m.name.end());
        out.insert(out.end(), m.bytes.begin(), m.bytes.end());
        central.push_back({m.name, m.crc, size, offset});
    }

    uint32_t cd_start = static_cast<uint32_t>(out.size());
    for (const auto& c : central) {
        put32(out, 0x02014b50);
        put16(out, 20); put16(out, 20);
        put16(out, 0); put16(out, 0);
        put16(out, 0); put16(out, 0);
        put32(out, c.crc);
        put32(out, c.size); put32(out, c.size);
        put16(out, static_cast<uint16_t>(c.name.size()));
        put16(out, 0); put16(out, 0);
        put16(out, 0); put16(out, 0);
        put32(out, 0);
        put32(out, c.offset);
        out.insert(out.end(), c.name.begin(), c.name.end());
    }
    uint32_t cd_size = static_cast<uint32_t>(out.size()) - cd_start;

    // End of central directory.
    put32(out, 0x06054b50);
    put16(out, 0); put16(out, 0);
    put16(out, static_cast<uint16_t>(central.size()));
    put16(out, static_cast<uint16_t>(central.size()));
    put32(out, cd_size);
    put32(out, cd_start);
    put16(out, 0);

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("Cannot open npz for writing: " + path);
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
}

}  // namespace npz
