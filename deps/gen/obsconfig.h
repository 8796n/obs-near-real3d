#pragma once
/*
 * Minimal stand-in for the CMake-generated obsconfig.h. OBS normally generates
 * this from libobs/obsconfig.h.in at its own build time; we only build a plugin
 * against the headers, so a concrete minimal version is enough. Values match a
 * standard Windows release install layout; the *_FOUND / ENABLE_WAYLAND knobs
 * are Linux-only and intentionally left undefined.
 */
#define OBS_DATA_PATH "../../data"
#define OBS_PLUGIN_PATH "../../obs-plugins/64bit"
#define OBS_PLUGIN_DESTINATION "obs-plugins/64bit"
#define OBS_RELEASE_CANDIDATE 0
#define OBS_BETA 0
