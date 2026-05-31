// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 8796n <info@8796.jp>
/*
 * near Real 3D (module obs-near-real3d) — monocular-depth 2D->3D side-by-side OBS video filter.
 *
 * Phase 2: real monocular depth via ONNX Runtime (DirectML), warped to a
 * side-by-side stereo pair by the GPU effect.
 *
 *   input frame  (process_filter -> texrender, recursion-safe)
 *     -> downscale + GPU->CPU stage -> ONNX depth on a worker thread
 *        (~30 fps, like the device's depth cadence; warp runs every frame)
 *     -> depth uploaded as a texture -> near-real3d.effect SBS warp -> screen
 *
 * Maps to: libnr_mono_depth.so (depth) + libnr_depth_manager.so (warp/Gen3D).
 */
#include <obs-module.h>
#include <graphics/vec2.h>
#include <util/platform.h>

#include <windows.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "depth_infer.hpp"
#include "flow_stabilizer.hpp"

/* Visible build stamp so the loaded DLL can be identified at a glance (shown in
 * the filter properties UI and logged on load). __DATE__/__TIME__ change every
 * rebuild, so a stale plugin is immediately obvious. The version is injected by
 * CMake (-DPLUGIN_VERSION, e.g. 0.1.<run_number> in CI); the fallback below is
 * only used for IDE/standalone builds that don't define it. */
#ifndef REAL3D_VERSION
#define REAL3D_VERSION "0.1.0-dev"
#endif
#define REAL3D_BUILD_INFO \
	("near Real 3D " REAL3D_VERSION "  (built " __DATE__ " " __TIME__ ")")

OBS_DECLARE_MODULE()
/* UI strings live in data/locale/<lang>.ini; default (and fallback) is en-US. */
OBS_MODULE_USE_DEFAULT_LOCALE("obs-near-real3d", "en-US")

MODULE_EXPORT const char *obs_module_description(void)
{
	return "near Real 3D: monocular-depth 2D->3D side-by-side filter";
}

static const float STRENGTH_FRAC[5] = {0.0f, 0.010f, 0.018f, 0.028f, 0.045f};
static const int INFER_SIZE = 392; /* must match the exported ONNX input size */
/* After motion stops, keep inferring this many "settle" frames so the temporal
 * blend converges to its ghost-free steady state before the static-skip freezes
 * the depth (otherwise a motion trail / post-cut ghost gets frozen in). */
static const int SETTLE_FRAMES = 8;
/* Mild Gaussian applied to the 392^2 inference input when "smooth input" is on,
 * to tame compression banding before depth inference (output is unaffected). */
static const float INPUT_SMOOTH_SIGMA = 0.8f;

static std::wstring utf8_to_wide(const char *s)
{
	if (!s)
		return L"";
	int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
	std::wstring w(n ? n - 1 : 0, L'\0');
	if (n)
		MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], n);
	return w;
}

struct real3d_filter {
	obs_source_t *context = nullptr;
	gs_effect_t *effect = nullptr;
	gs_eparam_t *p_image = nullptr, *p_strength = nullptr,
		    *p_conv = nullptr, *p_swap = nullptr,
		    *p_usedepth = nullptr, *p_depthtex = nullptr,
		    *p_showdepth = nullptr, *p_eyefit = nullptr;

	gs_texrender_t *rt_full = nullptr;  /* captured input at WxH */
	gs_texrender_t *rt_small_pre = nullptr; /* 2*INFER_SIZE area-avg stage */
	gs_texrender_t *rt_small = nullptr; /* downscaled to INFER_SIZE^2 */
	gs_stagesurf_t *stage = nullptr;    /* GPU->CPU readback */
	gs_texture_t *depth_tex = nullptr;  /* INFER_SIZE^2 R32F */

	/* tunables */
	float frac = 0.018f, convergence = 0.5f, swap_sign = 1.0f;
	bool full_sbs = true;
	int sbs_size = 0;          /* 0 = match source, 1 = 1080p (1920x1080/eye) */
	bool eye_letterbox = false; /* aspect-fit source into each eye (vs stretch) */
	bool show_depth = false;
	std::atomic<bool> input_smooth{true}; /* mild blur on inference input */
	bool logged_dims = false;
	uint64_t infer_interval_ns = 66666666ULL; /* depth cadence; 15 fps default */

	/* ONNX + flow-guided temporal stabiliser + worker */
	OrtDepth ort;
	FlowStabilizer flow;
	bool ort_ok = false;
	std::atomic<bool> ort_live{true}; /* false after a runtime Run() failure */
	std::thread worker;
	std::mutex m;
	std::condition_variable cv;
	std::vector<uint8_t> in_buf;   /* INFER_SIZE^2 * 4 RGBA, tight */
	std::vector<float> depth_buf;  /* INFER_SIZE^2 */
	bool input_ready = false, depth_ready = false, stop = false;
	bool logged_first_depth = false;
	uint64_t last_submit_ns = 0;

	/* static-frame skip: when the downscaled input barely changes we reuse
	 * the depth already in depth_tex instead of re-running ONNX */
	std::vector<uint8_t> prev_in; /* last *inferred* INFER_SIZE^2*4 frame */
	bool skip_static = true;
	float static_thresh = 1.0f;   /* mean abs RGB diff (0-255); 0 = exact */
	int settle_left = 0;          /* remaining settle inferences after motion */
};

static void worker_fn(real3d_filter *f)
{
	std::vector<uint8_t> local_in;
	std::vector<float> raw, stab;
	for (;;) {
		{
			std::unique_lock<std::mutex> lk(f->m);
			f->cv.wait(lk, [&] { return f->input_ready || f->stop; });
			if (f->stop)
				return;
			local_in.swap(f->in_buf);
			f->input_ready = false;
		}
		/* Mild blur of the inference input to tame compression banding. Run
		 * here on the worker (not the graphics thread) so it never stalls
		 * OBS's video_render; the visible warp uses the full-res frame, so
		 * output sharpness is unaffected. */
		if (f->input_smooth.load(std::memory_order_relaxed))
			nr3d::smoothRGBA(local_in.data(), f->ort.size(),
					 INPUT_SMOOTH_SIGMA);

		/* raw ONNX depth -> normalise + flow-guided temporal stabilise */
		if (f->ort.Run(local_in.data(), raw)) {
			f->flow.process(
				local_in.data(), f->ort.size(), raw, stab,
				f->ort.temporal.load(std::memory_order_relaxed),
				f->ort.stab_strength.load(std::memory_order_relaxed),
				f->ort.depth_smooth.load(std::memory_order_relaxed) * 6.0f);
			{
				std::lock_guard<std::mutex> lk(f->m);
				f->depth_buf.swap(stab);
				f->depth_ready = true;
			}
			f->ort_live.store(true, std::memory_order_relaxed);
		} else if (f->ort_live.exchange(false, std::memory_order_relaxed)) {
			/* runtime failure (vs the one-time Init failure): drop to the
			 * luminance-depth fallback instead of freezing the last depth. */
			blog(LOG_WARNING, "[near-real3d] ONNX inference failed (%s) "
					  "-> luminance-depth fallback",
			     f->ort.last_error().c_str());
		}
	}
}

static const char *real3d_get_name(void *)
{
	return "near Real 3D (SBS)";
}

static void real3d_update(void *data, obs_data_t *s)
{
	auto *f = static_cast<real3d_filter *>(data);
	long long tier = obs_data_get_int(s, "strength");
	if (tier < 0 || tier > 4)
		tier = 2;
	f->frac = STRENGTH_FRAC[tier];   /* tier 0 -> 0 disparity (flat, A/B) */
	f->convergence = (float)obs_data_get_double(s, "convergence");
	f->swap_sign = obs_data_get_bool(s, "swap") ? -1.0f : 1.0f;
	bool prev = f->full_sbs;
	f->full_sbs = obs_data_get_bool(s, "full_sbs");
	f->sbs_size = (int)obs_data_get_int(s, "sbs_size");
	f->eye_letterbox = obs_data_get_bool(s, "eye_letterbox");
	if (prev != f->full_sbs)
		f->logged_dims = false; /* re-log new output size once */

	long long fps = obs_data_get_int(s, "infer_fps");
	if (fps < 1 || fps > 60)
		fps = 15;
	f->infer_interval_ns = 1000000000ULL / (uint64_t)fps;

	f->skip_static = obs_data_get_bool(s, "skip_static");
	f->static_thresh = (float)obs_data_get_double(s, "static_thresh");

	f->ort.temporal.store(obs_data_get_bool(s, "temporal"),
			      std::memory_order_relaxed);
	f->ort.stab_strength.store((float)obs_data_get_double(s, "stabilize_strength"),
				   std::memory_order_relaxed);
	f->ort.depth_smooth.store((float)obs_data_get_double(s, "depth_smooth"),
				  std::memory_order_relaxed);
	f->show_depth = obs_data_get_bool(s, "show_depth");
	f->input_smooth.store(obs_data_get_bool(s, "input_smooth"),
			      std::memory_order_relaxed);
}

static void *real3d_create(obs_data_t *settings, obs_source_t *context)
{
	auto *f = new real3d_filter();
	f->context = context;

	char *effect_path = obs_module_file("near-real3d.effect");
	char *model_path = obs_module_file("depth_anything_v2_small.onnx");

	obs_enter_graphics();
	f->effect = gs_effect_create_from_file(effect_path, nullptr);
	if (f->effect) {
		f->p_image = gs_effect_get_param_by_name(f->effect, "image");
		f->p_strength = gs_effect_get_param_by_name(f->effect, "strength");
		f->p_conv = gs_effect_get_param_by_name(f->effect, "convergence");
		f->p_swap = gs_effect_get_param_by_name(f->effect, "swap_sign");
		f->p_usedepth = gs_effect_get_param_by_name(f->effect, "use_depth_tex");
		f->p_depthtex = gs_effect_get_param_by_name(f->effect, "depth_tex");
		f->p_showdepth = gs_effect_get_param_by_name(f->effect, "show_depth");
		f->p_eyefit = gs_effect_get_param_by_name(f->effect, "eye_fit");
	}
	f->rt_full = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	f->rt_small_pre = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	f->rt_small = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	f->stage = gs_stagesurface_create(INFER_SIZE, INFER_SIZE, GS_RGBA);
	std::vector<float> half(INFER_SIZE * INFER_SIZE, 0.5f);
	f->depth_tex = gs_texture_create(INFER_SIZE, INFER_SIZE, GS_R32F, 1,
					 nullptr, GS_DYNAMIC);
	gs_texture_set_image(f->depth_tex, (const uint8_t *)half.data(),
			     INFER_SIZE * sizeof(float), false);
	obs_leave_graphics();

	if (model_path) {
		f->ort_ok = f->ort.Init(utf8_to_wide(model_path), INFER_SIZE);
		if (!f->ort_ok)
			blog(LOG_WARNING, "[near-real3d] ONNX init failed (%s) "
					  "-> luminance-depth fallback",
			     f->ort.last_error().c_str());
	} else {
		blog(LOG_WARNING, "[near-real3d] model not found -> luminance-depth fallback");
	}

	bfree(effect_path);
	bfree(model_path);

	if (f->ort_ok)
		f->worker = std::thread(worker_fn, f);

	real3d_update(f, settings);
	blog(LOG_INFO, "[near-real3d] filter created (effect=%s, depth=%s)",
	     f->effect ? "ok" : "MISSING", f->ort_ok ? "onnx-dml" : "luminance");
	return f;
}

static void real3d_destroy(void *data)
{
	auto *f = static_cast<real3d_filter *>(data);
	if (f->worker.joinable()) {
		{
			std::lock_guard<std::mutex> lk(f->m);
			f->stop = true;
		}
		f->cv.notify_all();
		f->worker.join();
	}
	obs_enter_graphics();
	if (f->effect)
		gs_effect_destroy(f->effect);
	if (f->rt_full)
		gs_texrender_destroy(f->rt_full);
	if (f->rt_small_pre)
		gs_texrender_destroy(f->rt_small_pre);
	if (f->rt_small)
		gs_texrender_destroy(f->rt_small);
	if (f->stage)
		gs_stagesurface_destroy(f->stage);
	if (f->depth_tex)
		gs_texture_destroy(f->depth_tex);
	obs_leave_graphics();
	delete f;
}

static obs_properties_t *real3d_properties(void *)
{
	obs_properties_t *p = obs_properties_create();
	obs_property_t *q;

	/* ---- build identity (top of the panel) ---- */
	obs_properties_add_text(p, "build_info", REAL3D_BUILD_INFO,
				OBS_TEXT_INFO);

	/* ---- 3D ---- */
	obs_properties_t *g3d = obs_properties_create();
	obs_property_t *tier = obs_properties_add_list(
		g3d, "strength", obs_module_text("strength"),
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(tier, obs_module_text("strength.0"), 0);
	obs_property_list_add_int(tier, obs_module_text("strength.1"), 1);
	obs_property_list_add_int(tier, obs_module_text("strength.2"), 2);
	obs_property_list_add_int(tier, obs_module_text("strength.3"), 3);
	obs_property_list_add_int(tier, obs_module_text("strength.4"), 4);
	q = obs_properties_add_float_slider(g3d, "convergence",
					    obs_module_text("convergence"), 0.0,
					    1.0, 0.01);
	obs_property_set_long_description(q, obs_module_text("convergence.desc"));
	obs_properties_add_bool(g3d, "swap", obs_module_text("swap"));
	q = obs_properties_add_bool(g3d, "full_sbs", obs_module_text("fullsbs"));
	obs_property_set_long_description(q, obs_module_text("fullsbs.desc"));
	obs_property_t *ss = obs_properties_add_list(
		g3d, "sbs_size", obs_module_text("sbssize"), OBS_COMBO_TYPE_LIST,
		OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(ss, obs_module_text("sbssize.match"), 0);
	obs_property_list_add_int(ss, obs_module_text("sbssize.1080p"), 1);
	obs_property_set_long_description(ss, obs_module_text("sbssize.desc"));
	q = obs_properties_add_bool(g3d, "eye_letterbox",
				    obs_module_text("letterbox"));
	obs_property_set_long_description(q, obs_module_text("letterbox.desc"));
	obs_properties_add_group(p, "grp_3d", obs_module_text("group.3d"),
				 OBS_GROUP_NORMAL, g3d);

	/* ---- stabilization ---- */
	obs_properties_t *gst = obs_properties_create();
	obs_property_t *fps = obs_properties_add_list(
		gst, "infer_fps", obs_module_text("fps"), OBS_COMBO_TYPE_LIST,
		OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(fps, obs_module_text("fps.10"), 10);
	obs_property_list_add_int(fps, obs_module_text("fps.15"), 15);
	obs_property_list_add_int(fps, obs_module_text("fps.24"), 24);
	obs_property_list_add_int(fps, obs_module_text("fps.30"), 30);
	obs_property_list_add_int(fps, obs_module_text("fps.60"), 60);
	obs_property_set_long_description(fps, obs_module_text("fps.desc"));
	q = obs_properties_add_bool(gst, "skip_static",
				    obs_module_text("skipstatic"));
	obs_property_set_long_description(q, obs_module_text("skipstatic.desc"));
	q = obs_properties_add_float_slider(gst, "static_thresh",
					    obs_module_text("staticthresh"), 0.0,
					    8.0, 0.1);
	obs_property_set_long_description(q, obs_module_text("staticthresh.desc"));
	obs_properties_add_bool(gst, "temporal", obs_module_text("temporal"));
	q = obs_properties_add_float_slider(gst, "stabilize_strength",
					    obs_module_text("stabstrength"), 0.0,
					    1.0, 0.05);
	obs_property_set_long_description(q, obs_module_text("stabstrength.desc"));
	q = obs_properties_add_float_slider(gst, "depth_smooth",
					    obs_module_text("edgesoft"), 0.0, 1.0,
					    0.05);
	obs_property_set_long_description(q, obs_module_text("edgesoft.desc"));
	q = obs_properties_add_bool(gst, "input_smooth",
				    obs_module_text("inputsmooth"));
	obs_property_set_long_description(q, obs_module_text("inputsmooth.desc"));
	obs_properties_add_group(p, "grp_stab", obs_module_text("group.stab"),
				 OBS_GROUP_NORMAL, gst);

	/* ---- debug ---- */
	obs_properties_t *gdbg = obs_properties_create();
	q = obs_properties_add_bool(gdbg, "show_depth",
				    obs_module_text("showdepth"));
	obs_property_set_long_description(q, obs_module_text("showdepth.desc"));
	obs_properties_add_group(p, "grp_dbg", obs_module_text("group.debug"),
				 OBS_GROUP_NORMAL, gdbg);

	return p;
}

static void real3d_defaults(obs_data_t *s)
{
	obs_data_set_default_int(s, "strength", 2);
	obs_data_set_default_double(s, "convergence", 0.5);
	obs_data_set_default_bool(s, "swap", false);
	obs_data_set_default_bool(s, "full_sbs", true);
	obs_data_set_default_int(s, "sbs_size", 0);
	obs_data_set_default_bool(s, "eye_letterbox", false);
	obs_data_set_default_int(s, "infer_fps", 15);
	obs_data_set_default_bool(s, "skip_static", true);
	obs_data_set_default_double(s, "static_thresh", 1.0);
	obs_data_set_default_bool(s, "temporal", true);
	obs_data_set_default_double(s, "stabilize_strength", 0.4);
	obs_data_set_default_double(s, "depth_smooth", 0.3);
	obs_data_set_default_bool(s, "input_smooth", true);
	obs_data_set_default_bool(s, "show_depth", false);
}

/* Per-eye target resolution: match the source, or a fixed preset (e.g. 1080p so
 * the SBS frame fits glasses with per-eye Full HD regardless of source size). */
static void eye_dims(const real3d_filter *f, uint32_t srcW, uint32_t srcH,
		     uint32_t &eyeW, uint32_t &eyeH)
{
	if (f->sbs_size == 1) { eyeW = 1920; eyeH = 1080; }
	else { eyeW = srcW; eyeH = srcH; }
}

/* SBS layout changes the reported output width: Half-SBS keeps the eye width
 * (each eye squeezed into half), Full-SBS doubles it (full-res per eye). */
static uint32_t real3d_get_width(void *data)
{
	auto *f = static_cast<real3d_filter *>(data);
	obs_source_t *t = obs_filter_get_target(f->context);
	if (!t)
		return 0;
	uint32_t eyeW, eyeH;
	eye_dims(f, obs_source_get_base_width(t),
		 obs_source_get_base_height(t), eyeW, eyeH);
	return f->full_sbs ? eyeW * 2 : eyeW;
}

static uint32_t real3d_get_height(void *data)
{
	auto *f = static_cast<real3d_filter *>(data);
	obs_source_t *t = obs_filter_get_target(f->context);
	if (!t)
		return 0;
	uint32_t eyeW, eyeH;
	eye_dims(f, obs_source_get_base_width(t),
		 obs_source_get_base_height(t), eyeW, eyeH);
	return eyeH;
}

static void capture_input(real3d_filter *f, uint32_t w, uint32_t h,
			  enum gs_color_space space)
{
	gs_texrender_reset(f->rt_full);
	struct vec4 clr;
	vec4_zero(&clr);
	/* Capture in the source's color space so the sRGB decode that OBS does
	 * while drawing the source gets paired with an sRGB encode on store.
	 * Plain gs_texrender_begin() leaves the target non-sRGB, so the encode
	 * was a no-op and sRGB media ended up linearized (= darkened). */
	if (gs_texrender_begin_with_color_space(f->rt_full, w, h, space)) {
		gs_clear(GS_CLEAR_COLOR, &clr, 0.0f, 0);
		gs_ortho(0.0f, (float)w, 0.0f, (float)h, -100.0f, 100.0f);
		obs_source_process_filter_end(
			f->context, obs_get_base_effect(OBS_EFFECT_DEFAULT), w, h);
		gs_texrender_end(f->rt_full);
	} else {
		obs_source_process_filter_end(
			f->context, obs_get_base_effect(OBS_EFFECT_DEFAULT), w, h);
	}
}

/* True when `cur` is essentially identical to the last inferred frame `prev`:
 * mean per-channel RGB difference (subsampled) at or below `thresh`. Returns
 * false when there is no previous frame yet (sizes differ), so the very first
 * frame always runs inference. */
static bool frame_is_static(const std::vector<uint8_t> &cur,
			    const std::vector<uint8_t> &prev, float thresh)
{
	if (cur.empty() || cur.size() != prev.size())
		return false;
	const size_t n = cur.size();
	const size_t stride = 4 * 4; /* every 4th RGBA pixel is plenty */
	uint64_t sad = 0;
	size_t samples = 0;
	for (size_t i = 0; i + 4 <= n; i += stride) {
		for (int c = 0; c < 3; ++c) { /* R,G,B; alpha is constant */
			int d = (int)cur[i + c] - (int)prev[i + c];
			sad += (uint64_t)(d < 0 ? -d : d);
		}
		samples += 3;
	}
	float mad = samples ? (float)sad / (float)samples : 0.0f;
	return mad <= thresh;
}

static void maybe_submit_inference(real3d_filter *f, gs_texture_t *full)
{
	uint64_t now = os_gettime_ns();
	if (!f->ort_ok || now - f->last_submit_ns < f->infer_interval_ns)
		return;

	/* If the worker hasn't consumed the previous submission yet, skip this
	 * cadence entirely -- don't read back, don't advance the cadence, don't
	 * touch the settle budget. This throttles submission to the worker's real
	 * rate (so frames aren't dropped by overwriting in_buf when the requested
	 * fps exceeds worker throughput) and keeps settle_left counting inferences
	 * that actually run, not just submissions. */
	{
		std::lock_guard<std::mutex> lk(f->m);
		if (f->input_ready)
			return;
	}

	/* Two-stage area-average downscale (full -> 2*INFER_SIZE -> INFER_SIZE).
	 * A single large bilinear reduction under-samples (aliasing, and lets
	 * compression banding through); halving in two steps box-averages the
	 * footprint, giving the depth model a cleaner, de-banded input. Inference
	 * path only -- the visible warp still samples the full-res frame, so output
	 * sharpness is unaffected. */
	gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	gs_eparam_t *dimg = gs_effect_get_param_by_name(def, "image");
	struct vec4 clr;
	vec4_zero(&clr);
	const int PRE = INFER_SIZE * 2;

	gs_texrender_reset(f->rt_small_pre);
	if (!gs_texrender_begin(f->rt_small_pre, PRE, PRE))
		return;
	gs_clear(GS_CLEAR_COLOR, &clr, 0.0f, 0);
	gs_ortho(0.0f, (float)PRE, 0.0f, (float)PRE, -100.0f, 100.0f);
	gs_effect_set_texture(dimg, full);
	while (gs_effect_loop(def, "Draw"))
		gs_draw_sprite(full, 0, PRE, PRE);
	gs_texrender_end(f->rt_small_pre);
	gs_texture_t *pre = gs_texrender_get_texture(f->rt_small_pre);

	gs_texrender_reset(f->rt_small);
	if (!gs_texrender_begin(f->rt_small, INFER_SIZE, INFER_SIZE))
		return;
	gs_clear(GS_CLEAR_COLOR, &clr, 0.0f, 0);
	gs_ortho(0.0f, (float)INFER_SIZE, 0.0f, (float)INFER_SIZE, -100.0f, 100.0f);
	gs_effect_set_texture(dimg, pre);
	while (gs_effect_loop(def, "Draw"))
		gs_draw_sprite(pre, 0, INFER_SIZE, INFER_SIZE);
	gs_texrender_end(f->rt_small);

	gs_stage_texture(f->stage, gs_texrender_get_texture(f->rt_small));
	uint8_t *data = nullptr;
	uint32_t linesize = 0;
	if (gs_stagesurface_map(f->stage, &data, &linesize)) {
		std::vector<uint8_t> tight((size_t)INFER_SIZE * INFER_SIZE * 4);
		for (int y = 0; y < INFER_SIZE; ++y)
			memcpy(&tight[(size_t)y * INFER_SIZE * 4],
			       data + (size_t)y * linesize,
			       (size_t)INFER_SIZE * 4);
		gs_stagesurface_unmap(f->stage);

		/* One cadence tick is consumed whether or not we infer, so a
		 * static scene keeps doing the cheap readback at the depth rate
		 * but skips the expensive ONNX pass below. */
		f->last_submit_ns = now;

		/* Static-frame skip: reuse the depth already in depth_tex when this
		 * frame barely differs from the last one we inferred. But after motion
		 * stops we keep inferring for SETTLE_FRAMES so the temporal blend
		 * converges to its ghost-free steady state *before* freezing -- without
		 * this, a motion trail or post-cut ghost gets frozen in and never
		 * clears while inference is skipped. */
		const bool is_static =
			f->skip_static &&
			frame_is_static(tight, f->prev_in, f->static_thresh);
		if (is_static) {
			if (f->settle_left == 0)
				return; /* settled & static -> keep the frozen depth */
			f->settle_left--; /* still settling -> infer to converge */
		} else {
			f->settle_left = SETTLE_FRAMES; /* motion -> refill budget */
		}

		f->prev_in = tight; /* baseline for the next comparison */
		{
			std::lock_guard<std::mutex> lk(f->m);
			f->in_buf.swap(tight);
			f->input_ready = true;
		}
		f->cv.notify_one();
	}
}

static void real3d_video_render(void *data, gs_effect_t *)
{
	auto *f = static_cast<real3d_filter *>(data);
	obs_source_t *target = obs_filter_get_target(f->context);
	uint32_t w = target ? obs_source_get_base_width(target) : 0;
	uint32_t h = target ? obs_source_get_base_height(target) : 0;

	if (!f->effect || w == 0 || h == 0) {
		obs_source_skip_video_filter(f->context);
		return;
	}

	/* Resolve the source's color space (SDR -> GS_CS_SRGB) and thread it
	 * through both the filter capture and our intermediate texrender so the
	 * decode/encode pair up; otherwise sRGB media is left linearized and the
	 * output looks darkened. rt_full is 8-bit GS_RGBA, so we only advertise
	 * SDR sRGB as a supported space. */
	const enum gs_color_space pref[] = {GS_CS_SRGB};
	const enum gs_color_space space =
		obs_source_get_color_space(target, 1, pref);

	if (!obs_source_process_filter_begin_with_color_space(
		    f->context, GS_RGBA, space, OBS_NO_DIRECT_RENDERING))
		return;

	capture_input(f, w, h, space);
	gs_texture_t *full = gs_texrender_get_texture(f->rt_full);
	if (!full)
		return;

	maybe_submit_inference(f, full);

	/* upload any fresh depth (texture ops must be on the graphics thread) */
	{
		std::vector<float> got;
		{
			std::lock_guard<std::mutex> lk(f->m);
			if (f->depth_ready) {
				got.swap(f->depth_buf);
				f->depth_ready = false;
			}
		}
		if (!got.empty()) {
			gs_texture_set_image(f->depth_tex,
					     (const uint8_t *)got.data(),
					     INFER_SIZE * sizeof(float), false);
			if (!f->logged_first_depth) {
				f->logged_first_depth = true;
				blog(LOG_INFO, "[near-real3d] first ONNX depth "
					       "uploaded (%dx%d) - inference loop live",
				     INFER_SIZE, INFER_SIZE);
			}
		}
	}

	/* final SBS warp to the screen, sRGB-correct. Mirror OBS's own
	 * render_filter_tex(): when the canvas is in linear-light mode, sample
	 * the captured frame through its sRGB view (decode -> linear) and let the
	 * sRGB framebuffer re-encode on store. Plain set_texture left sRGB media
	 * linearized on screen (= the darkening). */
	const bool linear_srgb = gs_get_linear_srgb();
	const bool prev_fb = gs_framebuffer_srgb_enabled();
	gs_enable_framebuffer_srgb(linear_srgb);
	if (linear_srgb)
		gs_effect_set_texture_srgb(f->p_image, full);
	else
		gs_effect_set_texture(f->p_image, full);
	gs_effect_set_texture(f->p_depthtex, f->depth_tex);
	gs_effect_set_float(f->p_strength, f->frac);
	gs_effect_set_float(f->p_conv, f->convergence);
	gs_effect_set_float(f->p_swap, f->swap_sign);
	gs_effect_set_float(f->p_usedepth,
			    (f->ort_ok &&
			     f->ort_live.load(std::memory_order_relaxed))
				    ? 1.0f
				    : 0.0f);
	gs_effect_set_float(f->p_showdepth, f->show_depth ? 1.0f : 0.0f);

	/* per-eye target size + optional aspect-fit (letterbox) of the source */
	uint32_t eyeW, eyeH;
	eye_dims(f, w, h, eyeW, eyeH);
	struct vec2 efit;
	efit.x = 1.0f;
	efit.y = 1.0f;
	if (f->eye_letterbox && w && h && eyeW && eyeH) {
		const float src_a = (float)w / (float)h;
		const float eye_a = (float)eyeW / (float)eyeH;
		const float ratio = src_a / eye_a;
		if (ratio >= 1.0f)
			efit.y = 1.0f / ratio; /* source wider -> bars top/bottom */
		else
			efit.x = ratio;        /* source taller -> bars left/right */
	}
	gs_effect_set_vec2(f->p_eyefit, &efit);

	const uint32_t out_w = f->full_sbs ? eyeW * 2 : eyeW;
	const uint32_t out_h = eyeH;
	if (!f->logged_dims) {
		f->logged_dims = true;
		blog(LOG_INFO, "[near-real3d] output %ux%u (%s, source %ux%u) linear_srgb=%d",
		     out_w, out_h, f->full_sbs ? "Full-SBS" : "Half-SBS", w, h,
		     (int)linear_srgb);
	}
	while (gs_effect_loop(f->effect, "Draw"))
		gs_draw_sprite(full, 0, out_w, out_h);

	gs_enable_framebuffer_srgb(prev_fb);
}

static struct obs_source_info real3d_filter_info = {};

bool obs_module_load(void)
{
	real3d_filter_info.id = "near_real3d_sbs";
	real3d_filter_info.type = OBS_SOURCE_TYPE_FILTER;
	/* OBS_SOURCE_SRGB marks the filter sRGB-aware so libobs keeps the
	 * linear-light pipeline state consistent across our capture/draw passes
	 * (process_filter_tech_end sets gs_set_linear_srgb from this flag). */
	real3d_filter_info.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB;
	real3d_filter_info.get_name = real3d_get_name;
	real3d_filter_info.create = real3d_create;
	real3d_filter_info.destroy = real3d_destroy;
	real3d_filter_info.update = real3d_update;
	real3d_filter_info.get_properties = real3d_properties;
	real3d_filter_info.get_defaults = real3d_defaults;
	real3d_filter_info.get_width = real3d_get_width;
	real3d_filter_info.get_height = real3d_get_height;
	real3d_filter_info.video_render = real3d_video_render;
	obs_register_source(&real3d_filter_info);
	blog(LOG_INFO, "[near-real3d] loaded: %s (libobs %d.%d.%d)",
	     REAL3D_BUILD_INFO, LIBOBS_API_MAJOR_VER, LIBOBS_API_MINOR_VER,
	     LIBOBS_API_PATCH_VER);
	return true;
}

void obs_module_unload(void)
{
	blog(LOG_INFO, "[near-real3d] unloaded");
}
