cmake_minimum_required(VERSION 3.20)

# CMP0091: let CMAKE_MSVC_RUNTIME_LIBRARY pick the CRT. We must build /MT
# because the PhysX prebuilt static libs (PhysXExtensions_static_64.lib,
# PhysXPvdSDK_static_64.lib) under win.x86_64.vc143.mt are static-CRT.
cmake_policy(SET CMP0091 NEW)

project(ovminibrain LANGUAGES C CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT MSVC AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Supported platforms: Windows/MSVC and Linux")
endif()

# PhysX's PVD instrumentation is compiled OUT of the release libraries
# (PX_SUPPORT_PVD=0) and IN for the debug libraries (PX_SUPPORT_PVD=1).
# To allow PVD inspection we link the debug PhysX build in the Debug
# config and the release build otherwise. The debug PhysX libs use the
# debug static CRT (/MTd), the release libs the release static CRT (/MT),
# so the runtime must track the config. CMP0091 lets all FetchContent
# deps inherit this per-config CRT too.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# ----------------------------------------------------------------------
# PhysX 5.6.1 (locally built by the user)
# ----------------------------------------------------------------------
set(PHYSX_ROOT "C:/Git/PhysX/physx" CACHE PATH "Root of the PhysX 5.6.1 SDK")
set(PHYSX_INCLUDE_DIR "${PHYSX_ROOT}/include")
set(PHYSX_BIN_DIR "${PHYSX_ROOT}/bin/win.x86_64.vc143.mt")
set(PHYSX_LIB_DIR_RELEASE "${PHYSX_BIN_DIR}/release")
set(PHYSX_LIB_DIR_DEBUG   "${PHYSX_BIN_DIR}/debug")

if(NOT EXISTS "${PHYSX_INCLUDE_DIR}/PxPhysicsAPI.h")
    message(FATAL_ERROR "PhysX headers not found at ${PHYSX_INCLUDE_DIR}. Set -DPHYSX_ROOT=...")
endif()

# Static/import lib names (identical across configs). The Debug config
# pulls these from the PVD-enabled debug build; every other config from
# release. See the per-config linking below.
set(PHYSX_LIB_NAMES
    PhysX_64.lib
    PhysXCommon_64.lib
    PhysXFoundation_64.lib
    PhysXCooking_64.lib
    PhysXExtensions_static_64.lib
    PhysXPvdSDK_static_64.lib
)
# Runtime DLLs that must sit next to the exe (same names per config).
set(PHYSX_DLL_NAMES
    PhysX_64.dll
    PhysXCommon_64.dll
    PhysXFoundation_64.dll
    PhysXCooking_64.dll
)

# ----------------------------------------------------------------------
# ONNX Runtime 1.26.0 (C/C++ API). The conda env only ships the Python
# DLL (no headers/import lib), so fetch the official win-x64 release for
# headers + onnxruntime.lib + onnxruntime.dll. Override with -DORT_ROOT
# if you already have an unpacked onnxruntime-win-x64 tree.
# ----------------------------------------------------------------------
set(ORT_VERSION "1.26.0" CACHE STRING "ONNX Runtime version")
set(ORT_ROOT "" CACHE PATH "Optional: pre-unpacked onnxruntime-win-x64 root")

include(FetchContent)

if(WIN32)
    set(ORT_ARCHIVE "onnxruntime-win-x64-${ORT_VERSION}.zip")
else()
    set(ORT_ARCHIVE "onnxruntime-linux-x64-${ORT_VERSION}.tgz")
endif()

if(ORT_ROOT STREQUAL "")
    FetchContent_Declare(onnxruntime
        URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime)
    set(ORT_ROOT "${onnxruntime_SOURCE_DIR}")
endif()

# Locate the actual root containing include/onnxruntime_c_api.h (the archive
# may or may not strip its single top-level directory).
if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_c_api.h")
    file(GLOB_RECURSE _ort_hdr "${ORT_ROOT}/**/onnxruntime_c_api.h")
    if(_ort_hdr)
        list(GET _ort_hdr 0 _ort_hdr0)
        get_filename_component(_ort_inc "${_ort_hdr0}" DIRECTORY)         # .../include
        get_filename_component(ORT_ROOT "${_ort_inc}" DIRECTORY)         # .../<root>
    else()
        message(FATAL_ERROR "Could not find onnxruntime_c_api.h under ${ORT_ROOT}")
    endif()
endif()

set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB "${ORT_ROOT}/lib/onnxruntime.lib")
set(ORT_DLL "${ORT_ROOT}/lib/onnxruntime.dll")
if(NOT WIN32)
    set(ORT_LIB "${ORT_ROOT}/lib/libonnxruntime.so")
endif()

# ----------------------------------------------------------------------
# pugixml (MJCF parsing) + yaml-cpp (sidecar YAML). Both are CMake
# projects and inherit CMAKE_MSVC_RUNTIME_LIBRARY (CMP0091).
# ----------------------------------------------------------------------
FetchContent_Declare(pugixml
    GIT_REPOSITORY https://github.com/zeux/pugixml.git
    GIT_TAG v1.14
)
FetchContent_MakeAvailable(pugixml)

set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG 0.8.0
)
FetchContent_MakeAvailable(yaml-cpp)

# ----------------------------------------------------------------------
# Renderer deps: GLFW 3.4 (window/input, static), glm (header-only math),
# glad v2 (GL 3.3 core loader, generated at configure time via Python).
# All static, so no extra runtime DLLs. They inherit the /MT runtime
# (CMP0091) like the other FetchContent targets.
# ----------------------------------------------------------------------
if(UNIX)
    set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
    find_package(OpenGL REQUIRED)
    find_package(Threads REQUIRED)
endif()
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v2.0.8
    SOURCE_SUBDIR cmake
)
FetchContent_MakeAvailable(glad)
glad_add_library(glad_gl_core_33 REPRODUCIBLE API gl:core=3.3)

# ----------------------------------------------------------------------
# The app
# ----------------------------------------------------------------------
add_executable(ovminibrain
    src/main.cpp
    src/Config.cpp
    src/PhysxWorld.cpp
    src/SmplArticulation.cpp
    src/CollisionPrimitives.cpp
    src/ProjectilePool.cpp
    src/MotionCache.cpp
    src/ObsBuilder.cpp
    src/OnnxPolicy.cpp
    src/RobotChain.cpp
    src/ParityDump.cpp
    src/MjcfModel.cpp
    src/Renderer.cpp
)

target_include_directories(ovminibrain PRIVATE
    "${PHYSX_INCLUDE_DIR}"
    "${ORT_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

# Link the debug PhysX libs (PVD enabled) in Debug, release libs otherwise.
# CMake's debug/optimized keywords map to the Debug config vs the rest.
if(WIN32)
    foreach(lib ${PHYSX_LIB_NAMES})
        target_link_libraries(ovminibrain PRIVATE
            debug "${PHYSX_LIB_DIR_DEBUG}/${lib}"
            optimized "${PHYSX_LIB_DIR_RELEASE}/${lib}")
    endforeach()
else()
    set(PHYSX_LINUX_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/physx-sdk/bin/linux.x86_64/release" CACHE PATH "PhysX Linux static libraries")
    target_link_libraries(ovminibrain PRIVATE "-Wl,--start-group")
    foreach(lib PhysXExtensions PhysX PhysXPvdSDK PhysXCooking PhysXCommon PhysXFoundation)
        target_link_libraries(ovminibrain PRIVATE "${PHYSX_LINUX_LIB_DIR}/lib${lib}_static.a")
    endforeach()
    target_link_libraries(ovminibrain PRIVATE "-Wl,--end-group" Threads::Threads ${CMAKE_DL_LIBS} OpenGL::GL)
    target_compile_definitions(ovminibrain PRIVATE PX_PHYSX_STATIC_LIB)
endif()

target_link_libraries(ovminibrain PRIVATE
    "${ORT_LIB}"
    pugixml::pugixml
    yaml-cpp::yaml-cpp
    glfw
    glad_gl_core_33
    glm::glm
    $<$<PLATFORM_ID:Windows>:opengl32>
)

if(UNIX)
    set_target_properties(ovminibrain PROPERTIES BUILD_RPATH "$ORIGIN/../lib")
endif()

if(BUILD_TESTING)
    add_executable(test_minibrain_inputs tests/test_minibrain_inputs.cpp src/ObsBuilder.cpp src/Config.cpp)
    target_include_directories(test_minibrain_inputs PRIVATE src "${PHYSX_INCLUDE_DIR}")
    target_link_libraries(test_minibrain_inputs PRIVATE yaml-cpp::yaml-cpp)
    add_test(NAME minibrain_input_mapping COMMAND test_minibrain_inputs)
    foreach(clip idle left_jab)
        add_test(NAME minibrain_${clip} COMMAND ovminibrain
            --onnx "${CMAKE_CURRENT_SOURCE_DIR}/models/smpl/unified_pipeline.onnx"
            --motion-cache "${CMAKE_CURRENT_SOURCE_DIR}/motion_cache/${clip}.pmc"
            --mjcf "${CMAKE_CURRENT_SOURCE_DIR}/assets/smpl_humanoid.xml"
            --no-randomize-stamina --max-steps 60)
    endforeach()
    add_test(NAME minibrain_batch COMMAND ovminibrain
        --onnx "${CMAKE_CURRENT_SOURCE_DIR}/models/smpl/unified_pipeline.onnx"
        --motion-cache "${CMAKE_CURRENT_SOURCE_DIR}/motion_cache/left_jab.pmc"
        --mjcf "${CMAKE_CURRENT_SOURCE_DIR}/assets/smpl_humanoid.xml"
        --num-characters 4 --max-steps 120 --stamina-range 0.2 1.0)
endif()

# PhysX shared config does not define PX_PHYSX_STATIC_LIB. The Debug config
# links the debug PhysX build, so it must match those libs' defines
# (PX_DEBUG/PX_CHECKED) for ABI compatibility, and enables our PVD code.
target_compile_definitions(ovminibrain PRIVATE
    NOMINMAX WIN32_LEAN_AND_MEAN
    $<$<CONFIG:Debug>:PX_DEBUG=1;PX_CHECKED=1;OVPP_ENABLE_PVD=1>)

if(WIN32)
# Copy the matching per-config PhysX runtime DLLs next to the exe.
foreach(dll ${PHYSX_DLL_NAMES})
    add_custom_command(TARGET ovminibrain POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PHYSX_BIN_DIR}/$<IF:$<CONFIG:Debug>,debug,release>/${dll}"
            "$<TARGET_FILE_DIR:ovminibrain>"
        VERBATIM
    )
endforeach()

# ONNX Runtime ships a single (release) DLL with its own CRT; copy as-is.
add_custom_command(TARGET ovminibrain POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_DLL}" "$<TARGET_FILE_DIR:ovminibrain>"
    VERBATIM
)

endif()
