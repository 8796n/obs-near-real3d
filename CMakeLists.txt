cmake_minimum_required(VERSION 3.20)
project(obs-near-real3d VERSION 0.3.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- libobs (headers from cloned obs-studio 32.1.2, import lib from obs.dll) ---
set(OBS_SRC "${CMAKE_SOURCE_DIR}/deps/obs-studio")
set(OBS_LIB "${CMAKE_SOURCE_DIR}/deps/obslib/obs.lib")

# --- ONNX Runtime (DirectML EP) ---
set(ORT "${CMAKE_SOURCE_DIR}/deps/onnxruntime")

add_library(obs-near-real3d MODULE)
target_sources(obs-near-real3d PRIVATE src/plugin-main.cpp)

target_include_directories(obs-near-real3d PRIVATE
  "${OBS_SRC}/libobs"
  "${CMAKE_SOURCE_DIR}/deps/gen"   # minimal generated obsconfig.h
  "${ORT}/include")
target_link_libraries(obs-near-real3d PRIVATE
  "${OBS_LIB}"
  "${ORT}/lib/onnxruntime.lib")

# OBS modules are plain DLLs named exactly <id>.dll (no "lib" prefix).
set_target_properties(obs-near-real3d PROPERTIES PREFIX "" OUTPUT_NAME "obs-near-real3d")

# Version string baked into the binary (shown in the filter UI + logs). CI
# overrides it for releases, e.g. -DPLUGIN_VERSION=v0.3.0 for a tag build;
# local/IDE builds fall back to the project version with a -dev suffix.
if(NOT DEFINED PLUGIN_VERSION OR PLUGIN_VERSION STREQUAL "")
  set(PLUGIN_VERSION "${PROJECT_VERSION}-dev")
endif()
target_compile_definitions(obs-near-real3d PRIVATE REAL3D_VERSION="${PLUGIN_VERSION}")
message(STATUS "near Real 3D version: ${PLUGIN_VERSION}")

if(MSVC)
  target_compile_definitions(obs-near-real3d PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
  target_compile_options(obs-near-real3d PRIVATE /utf-8 /EHsc)
endif()
