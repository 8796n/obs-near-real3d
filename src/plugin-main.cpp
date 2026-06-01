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

/* ---- scene-cut handling (graphics thread) ----
 * The image updates every render frame but the depth lags by the inference
 * latency, so at a hard cut the new image would be warped by the previous
 * scene's stale depth (= a momentary parallax glitch). We sample a small
 * readback every render frame to detect the cut, pin the disparity flat (pure
 * 2D) so nothing is warped by stale depth, and fire an off-cadence inference so
 * fresh depth lands ASAP; the disparity then ramps back once the post-cut depth
 * arrives (paired by generation tag). */
static const float CUT_MAD = 30.0f;        /* mean abs RGB diff for a hard cut (0..255) */
/* Cut detection rides the canvas frame rate (video_render fires once per output
 * frame); this caps it so 120/144 fps canvases don't over-read. <=60 fps passes
 * through unthrottled (16.6ms frame gap > 13.8ms here), higher rates clamp ~72. */
static const uint64_t DETECT_MIN_INTERVAL_NS = 1000000000ULL / 72;
static const float DISP_RAMP_PER_SEC = 6.0f; /* disparity fade-in after a cut (~0.17 s) */
/* Failsafe: release the flat hold even if no post-cut depth ever arrives (e.g.
 * inference failure / worker stall) so the filter can't get stuck flat. */
static const uint64_t FLATTEN_TIMEOUT_NS = 500000000ULL;

/* ---- frame-matched delay mode (optional) ----
 * The depth lags the image by the inference pipeline latency, so by default the
 * warp applies slightly-stale depth to the current frame. In this mode we hold a
 * ring of recent full-res frames and warp the one that *matches* the depth we
 * currently have (latency measured via a capture-timestamp round-trip through
 * the worker), removing the constant-latency part of the misalignment. Cost:
 * the displayed video is delayed by that latency (~0.1 s) and the frame ring
 * uses extra VRAM. Audio is delayed to match via the source's sync offset. */
static const int DELAY_RING_MAX = 16;            /* cap on buffered frames (VRAM bound) */
static const float DELAY_EMA = 0.85f;            /* latency low-pass (per inference) */
/* The committed delay (frames) drives BOTH the video ring and the audio sync
 * offset. Video can change cheaply every frame, but changing the audio offset
 * makes OBS re-time the audio (an audible click), so we only re-commit when the
 * measured latency leaves a deadband around the current value AND a cooldown has
 * passed -- keeping the offset rock-steady in the steady state. */
static const float COMMIT_DEADBAND = 0.75f;      /* frames: ignore sub-frame jitter */
static const uint64_t COMMIT_COOLDOWN_NS = 1500000000ULL; /* 1.5 s between re-commits */
static_assert(COMMIT_DEADBAND > 0.5f, "deadband must exceed the rounding boundary");

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
	gs_stagesurf_t *stage[2] = {nullptr, nullptr}; /* GPU->CPU readback, ping-pong */
	int stage_cur = 0;             /* surface staged this tick; map the other */
	bool stage_primed = false;     /* false until both surfaces hold a frame */
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
	uint32_t in_gen = 0;           /* (m) scene-cut generation of in_buf's frame */
	uint32_t out_gen = 0;          /* (m) generation depth_buf was computed for */
	uint64_t in_ts = 0;            /* (m) capture timestamp of in_buf's frame (ns) */
	uint64_t out_ts = 0;           /* (m) capture timestamp depth_buf was computed for */
	bool logged_first_depth = false;
	uint64_t last_submit_ns = 0;

	/* static-frame skip: when the downscaled input barely changes we reuse
	 * the depth already in depth_tex instead of re-running ONNX */
	std::vector<uint8_t> prev_in; /* last *inferred* INFER_SIZE^2*4 frame */
	bool skip_static = true;
	float static_thresh = 1.0f;   /* mean abs RGB diff (0-255); 0 = exact */
	int settle_left = 0;          /* remaining settle inferences after motion */

	/* scene-cut handling (graphics thread only, except in_gen/out_gen under m) */
	uint64_t last_detect_ns = 0;  /* cut-detection cadence (canvas fps, capped) */
	std::vector<uint8_t> prev_detect; /* previous frame sampled for cut detection */
	uint32_t scene_gen = 0;       /* ++ on each detected cut */
	uint32_t awaiting_gen = 0;    /* release the flat hold once out_gen reaches this */
	bool hold_flat = false;       /* disparity pinned to 0 (2D) until post-cut depth */
	float disp_scale = 1.0f;      /* 0..1 multiplier on frac; ramps back after a cut */
	uint64_t cut_ns = 0;          /* when the current flat hold started (failsafe) */
	uint64_t last_render_ns = 0;  /* previous render timestamp, for fps-independent ramp */

	/* frame-matched delay mode */
	std::atomic<bool> sync_delay{false}; /* enable image-delay + audio sync */
	uint64_t cur_capture_ns = 0;  /* capture timestamp of this render's frame */
	uint64_t staged_ts[2] = {0, 0}; /* capture ts paired with each staging surface */
	double delay_ema_ns = 0.0;    /* measured pipeline latency (capture->depth), LPF */
	bool delay_ema_init = false;
	int commit_delay = -1;        /* committed delay in frames (drives video+audio); -1 = unset */
	uint64_t last_commit_ns = 0;  /* throttles re-commits so audio offset stays steady */
	std::vector<gs_texture_t *> ring; /* recent full-res frames (delay line) */
	int ring_w = 0, ring_h = 0;   /* ring slot dims (rebuild on source resize) */
	int ring_widx = 0;            /* next slot to write */
	int ring_filled = 0;          /* slots written so far (for warm-up) */
	/* audio sync via the source's sync offset (OBS buffers the audio for us) */
	bool sync_owned = false;      /* we currently manage the parent's sync offset */
	int64_t saved_sync = 0;       /* user's sync offset, restored on disable/destroy */
	int64_t applied_extra = -1;   /* audio delay we last applied (ns); -1 = none */
};

/* Destroy the delay ring. Caller must hold the graphics context. */
static void free_delay_ring(real3d_filter *f)
{
	for (gs_texture_t *t : f->ring)
		if (t)
			gs_texture_destroy(t);
	f->ring.clear();
	f->ring_w = f->ring_h = 0;
	f->ring_widx = f->ring_filled = 0;
}

/* Ensure the ring has `slots` textures at WxH, rebuilding on resize / growth.
 * Caller must hold the graphics context. Grows only (capped) to avoid churn. */
static void ensure_delay_ring(real3d_filter *f, uint32_t w, uint32_t h, int slots)
{
	if (slots < 2)
		slots = 2;
	if (slots > DELAY_RING_MAX)
		slots = DELAY_RING_MAX;
	const bool dims_ok = (f->ring_w == (int)w && f->ring_h == (int)h);
	if (dims_ok && (int)f->ring.size() >= slots)
		return; /* already big enough at the right size */
	free_delay_ring(f);
	f->ring.resize((size_t)slots, nullptr);
	for (int i = 0; i < slots; ++i)
		f->ring[i] = gs_texture_create(w, h, GS_RGBA, 1, nullptr,
					       GS_RENDER_TARGET);
	f->ring_w = (int)w;
	f->ring_h = (int)h;
	f->ring_widx = 0;
	f->ring_filled = 0;
}

/* Restore the parent source's audio sync offset we took over. Safe to call when
 * we don't own it (no-op). */
static void release_audio_sync(real3d_filter *f)
{
	if (!f->sync_owned)
		return;
	obs_source_t *parent = obs_filter_get_parent(f->context);
	if (parent)
		obs_source_set_sync_offset(parent, f->saved_sync);
	f->sync_owned = false;
	f->applied_extra = -1;
	f->commit_delay = -1;
}

static void worker_fn(real3d_filter *f)
{
	std::vector<uint8_t> local_in;
	std::vector<uint8_t> flow_in;
	std::vector<float> raw, stab;
	for (;;) {
		uint32_t gen = 0;
		uint64_t ts = 0;
		{
			std::unique_lock<std::mutex> lk(f->m);
			f->cv.wait(lk, [&] { return f->input_ready || f->stop; });
			if (f->stop)
				return;
			local_in.swap(f->in_buf);
			gen = f->in_gen; /* which cut generation this frame belongs to */
			ts = f->in_ts;   /* capture timestamp, for the delay-mode latency measure */
			f->input_ready = false;
		}
		/* Mild blur of the inference input to tame compression banding. Run
		 * here on the worker (not the graphics thread) so it never stalls
		 * OBS's video_render; the visible warp uses the full-res frame, so
		 * output sharpness is unaffected. */
		const bool temporal = f->ort.temporal.load(std::memory_order_relaxed);
		const TemporalMode temporal_mode =
			(TemporalMode)f->ort.temporal_mode.load(std::memory_order_relaxed);
		const bool input_smooth =
			f->input_smooth.load(std::memory_order_relaxed);
		/* Keep sharp luma for the experimental modes' flow / reactive-mask
		 * decisions. Legacy intentionally retains the release behaviour. The
		 * model still receives the smoothed copy to suppress compression noise. */
		if (temporal && temporal_mode != TemporalMode::Legacy && input_smooth)
			flow_in = local_in;
		else
			flow_in.clear();
		if (input_smooth)
			nr3d::smoothRGBA(local_in.data(), f->ort.size(),
					 INPUT_SMOOTH_SIGMA);

		/* raw ONNX depth -> normalise + flow-guided temporal stabilise */
		if (f->ort.Run(local_in.data(), raw)) {
			f->flow.process(
				flow_in.empty() ? local_in.data() : flow_in.data(),
				f->ort.size(), raw, stab, temporal, temporal_mode,
				f->ort.stab_strength.load(std::memory_order_relaxed),
				f->ort.depth_smooth.load(std::memory_order_relaxed) * 6.0f);
			{
				std::lock_guard<std::mutex> lk(f->m);
				f->depth_buf.swap(stab);
				f->out_gen = gen; /* tag the result with its cut generation */
				f->out_ts = ts;   /* ...and the frame's capture timestamp */
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
	long long temporal_mode = obs_data_get_int(s, "temporal_mode");
	if (temporal_mode < (long long)TemporalMode::Legacy ||
	    temporal_mode > (long long)TemporalMode::ReactiveClip)
		temporal_mode = (long long)TemporalMode::Legacy;
	f->ort.temporal_mode.store((int)temporal_mode, std::memory_order_relaxed);
	f->ort.stab_strength.store((float)obs_data_get_double(s, "stabilize_strength"),
				   std::memory_order_relaxed);
	f->ort.depth_smooth.store((float)obs_data_get_double(s, "depth_smooth"),
				  std::memory_order_relaxed);
	f->show_depth = obs_data_get_bool(s, "show_depth");
	f->input_smooth.store(obs_data_get_bool(s, "input_smooth"),
			      std::memory_order_relaxed);
	/* Frame-matched delay: just record intent here; acquiring/releasing the
	 * parent's audio sync offset and (re)building the frame ring happen on the
	 * graphics thread in real3d_video_render, where the parent is always valid. */
	f->sync_delay.store(obs_data_get_bool(s, "sync_delay"),
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
	f->stage[0] = gs_stagesurface_create(INFER_SIZE, INFER_SIZE, GS_RGBA);
	f->stage[1] = gs_stagesurface_create(INFER_SIZE, INFER_SIZE, GS_RGBA);
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
	release_audio_sync(f); /* hand the source's sync offset back to the user */
	if (f->worker.joinable()) {
		{
			std::lock_guard<std::mutex> lk(f->m);
			f->stop = true;
		}
		f->cv.notify_all();
		f->worker.join();
	}
	obs_enter_graphics();
	free_delay_ring(f);
	if (f->effect)
		gs_effect_destroy(f->effect);
	if (f->rt_full)
		gs_texrender_destroy(f->rt_full);
	if (f->rt_small_pre)
		gs_texrender_destroy(f->rt_small_pre);
	if (f->rt_small)
		gs_texrender_destroy(f->rt_small);
	if (f->stage[0])
		gs_stagesurface_destroy(f->stage[0]);
	if (f->stage[1])
		gs_stagesurface_destroy(f->stage[1]);
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
	obs_property_t *tm = obs_properties_add_list(
		gst, "temporal_mode", obs_module_text("temporalmode"),
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(tm, obs_module_text("temporalmode.legacy"),
				  (long long)TemporalMode::Legacy);
	obs_property_list_add_int(tm, obs_module_text("temporalmode.clip"),
				  (long long)TemporalMode::HistoryClip);
	obs_property_list_add_int(tm, obs_module_text("temporalmode.reactive"),
				  (long long)TemporalMode::ReactiveClip);
	obs_property_set_long_description(tm, obs_module_text("temporalmode.desc"));
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
	q = obs_properties_add_bool(gst, "sync_delay",
				    obs_module_text("syncdelay"));
	obs_property_set_long_description(q, obs_module_text("syncdelay.desc"));
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
	obs_data_set_default_int(s, "temporal_mode", (long long)TemporalMode::Legacy);
	obs_data_set_default_double(s, "stabilize_strength", 0.4);
	obs_data_set_default_double(s, "depth_smooth", 0.3);
	obs_data_set_default_bool(s, "input_smooth", true);
	obs_data_set_default_bool(s, "sync_delay", false);
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

/* Mean per-channel RGB difference (subsampled) between two equal-size INFER
 * frames; -1 when there is no comparable previous frame yet (sizes differ).
 * Shared by the static-skip and the scene-cut detection. */
static float frame_mad(const std::vector<uint8_t> &cur,
		       const std::vector<uint8_t> &prev)
{
	if (cur.empty() || cur.size() != prev.size())
		return -1.0f;
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
	return samples ? (float)sad / (float)samples : 0.0f;
}

/* True when `cur` is essentially identical to the last inferred frame `prev`:
 * mean per-channel RGB difference at or below `thresh`. False when there is no
 * previous frame yet, so the very first frame always runs inference. */
static bool frame_is_static(const std::vector<uint8_t> &cur,
			    const std::vector<uint8_t> &prev, float thresh)
{
	const float mad = frame_mad(cur, prev);
	return mad >= 0.0f && mad <= thresh;
}

static void maybe_submit_inference(real3d_filter *f, gs_texture_t *full)
{
	uint64_t now = os_gettime_ns();
	/* Cut detection rides the canvas frame rate (this callback fires once per
	 * output frame), capped so very high canvas rates don't over-read. The
	 * expensive ONNX submission below still honours the separate infer cadence,
	 * except a detected cut bypasses it to refresh the depth as fast as possible. */
	if (!f->ort_ok || now - f->last_detect_ns < DETECT_MIN_INTERVAL_NS)
		return;
	f->last_detect_ns = now;

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

	/* Double-buffered GPU->CPU readback: stage this frame into one surface and
	 * map the OTHER (staged last tick), so the map never waits on the just-issued
	 * copy -- no per-frame pipeline stall even at 60 fps. The mapped frame is one
	 * detect-tick old, which only adds ~1 frame of cut-detection latency. */
	const int cur = f->stage_cur;
	gs_stage_texture(f->stage[cur], gs_texrender_get_texture(f->rt_small));
	f->staged_ts[cur] = f->cur_capture_ns; /* pair this readback with its capture time */
	f->stage_cur ^= 1;
	if (!f->stage_primed) {
		f->stage_primed = true; /* other surface not staged yet -> wait one tick */
		return;
	}
	gs_stagesurf_t *readback = f->stage[cur ^ 1];
	const uint64_t readback_ts = f->staged_ts[cur ^ 1]; /* capture ts of the mapped frame */

	uint8_t *data = nullptr;
	uint32_t linesize = 0;
	if (!gs_stagesurface_map(readback, &data, &linesize))
		return;
	std::vector<uint8_t> tight((size_t)INFER_SIZE * INFER_SIZE * 4);
	for (int y = 0; y < INFER_SIZE; ++y)
		memcpy(&tight[(size_t)y * INFER_SIZE * 4],
		       data + (size_t)y * linesize,
		       (size_t)INFER_SIZE * 4);
	gs_stagesurface_unmap(readback);

	/* Hard-cut detection: large mean abs RGB diff vs the previous sampled frame.
	 * On a cut we (1) pin the disparity flat (real3d_video_render) so the new
	 * image isn't warped by the previous scene's now-stale depth, and (2) force an
	 * off-cadence inference -- overwriting any queued frame -- so fresh depth lands
	 * ASAP. The flat hold ends only when a depth tagged at/after this cut arrives
	 * (generation handshake), so a stale in-flight pre-cut result can't end it. */
	const float cut_mad = frame_mad(tight, f->prev_detect);
	const bool cut = cut_mad >= 0.0f && cut_mad >= CUT_MAD;
	f->prev_detect = tight;

	if (cut) {
		f->scene_gen++;
		f->awaiting_gen = f->scene_gen;
		f->hold_flat = true;
		f->disp_scale = 0.0f;
		f->cut_ns = now;
		f->settle_left = SETTLE_FRAMES;
		f->prev_in = tight; /* fresh static-skip baseline for the new scene */
		f->last_submit_ns = now;
		{
			std::lock_guard<std::mutex> lk(f->m);
			f->in_buf = tight; /* overwrite any unconsumed queued (stale) frame */
			f->in_gen = f->scene_gen;
			f->in_ts = readback_ts;
			f->input_ready = true;
		}
		f->cv.notify_one();
		return;
	}

	/* No cut: honour the infer cadence and the worker's real throughput. The
	 * readback above still ran at the detect rate, so cuts are caught between
	 * inferences; here we only gate the expensive ONNX submission. */
	if (now - f->last_submit_ns < f->infer_interval_ns)
		return;
	{
		std::lock_guard<std::mutex> lk(f->m);
		if (f->input_ready)
			return; /* worker hasn't consumed the last submission yet */
	}

	/* One cadence tick is consumed whether or not we infer, so a static scene
	 * keeps doing the cheap readback at the depth rate but skips the ONNX pass. */
	f->last_submit_ns = now;

	/* Static-frame skip: reuse the depth already in depth_tex when this frame
	 * barely differs from the last one we inferred. But after motion stops we keep
	 * inferring for SETTLE_FRAMES so the temporal blend converges to its ghost-free
	 * steady state *before* freezing -- without this, a motion trail or post-cut
	 * ghost gets frozen in and never clears while inference is skipped. */
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
		f->in_gen = f->scene_gen;
		f->in_ts = readback_ts;
		f->input_ready = true;
	}
	f->cv.notify_one();
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

	/* One timestamp for the whole render call: the captured frame's time (for
	 * delay-mode latency), the depth-arrival time, and the ramp dt. */
	const uint64_t now = os_gettime_ns();
	f->cur_capture_ns = now;

	capture_input(f, w, h, space);
	gs_texture_t *full = gs_texrender_get_texture(f->rt_full);
	if (!full)
		return;

	maybe_submit_inference(f, full);

	/* upload any fresh depth (texture ops must be on the graphics thread) */
	{
		std::vector<float> got;
		uint32_t got_gen = 0;
		uint64_t got_ts = 0;
		{
			std::lock_guard<std::mutex> lk(f->m);
			if (f->depth_ready) {
				got.swap(f->depth_buf);
				got_gen = f->out_gen;
				got_ts = f->out_ts;
				f->depth_ready = false;
			}
		}
		if (!got.empty()) {
			gs_texture_set_image(f->depth_tex,
					     (const uint8_t *)got.data(),
					     INFER_SIZE * sizeof(float), false);
			/* A depth computed at/after the latest cut means the warp can
			 * trust it again: release the flat hold so the disparity ramps
			 * back in. A stale pre-cut result (got_gen < awaiting) is uploaded
			 * but stays invisible while the disparity is still pinned flat. */
			if (f->hold_flat && got_gen >= f->awaiting_gen)
				f->hold_flat = false;
			/* Measure the capture->depth pipeline latency (LPF). This is how
			 * far the image must be delayed in sync mode to match the depth. */
			if (got_ts && now > got_ts) {
				const double lat = (double)(now - got_ts);
				if (!f->delay_ema_init) {
					f->delay_ema_ns = lat;
					f->delay_ema_init = true;
				} else {
					f->delay_ema_ns =
						DELAY_EMA * f->delay_ema_ns +
						(1.0 - DELAY_EMA) * lat;
				}
			}
			if (!f->logged_first_depth) {
				f->logged_first_depth = true;
				blog(LOG_INFO, "[near-real3d] first ONNX depth "
					       "uploaded (%dx%d) - inference loop live",
				     INFER_SIZE, INFER_SIZE);
			}
		}
	}

	/* Scene-cut disparity fade: held at 0 (flat/2D) while waiting for the
	 * post-cut depth, then ramped back to full over ~1/DISP_RAMP_PER_SEC s so
	 * the 3D re-appears without a hard pop. fps-independent via the render dt.
	 * A failsafe releases the hold if fresh depth never arrives (e.g. inference
	 * failure) so the filter can't get stuck flat. */
	{
		if (f->hold_flat && now - f->cut_ns > FLATTEN_TIMEOUT_NS)
			f->hold_flat = false;
		const float dt = f->last_render_ns
					 ? (float)(now - f->last_render_ns) * 1e-9f
					 : 0.0f;
		f->last_render_ns = now;
		if (f->hold_flat) {
			f->disp_scale = 0.0f;
		} else if (f->disp_scale < 1.0f) {
			f->disp_scale += DISP_RAMP_PER_SEC * dt;
			if (f->disp_scale > 1.0f)
				f->disp_scale = 1.0f;
		}
	}

	/* Frame-matched delay: pick which buffered frame to warp so it lines up with
	 * the depth we currently have, and delay the source audio to match. When the
	 * mode is off, release the audio sync and free the ring. */
	gs_texture_t *show = full;
	{
		struct obs_video_info ovi;
		uint64_t interval_ns = 0;
		if (obs_get_video_info(&ovi) && ovi.fps_num)
			interval_ns = (uint64_t)ovi.fps_den * 1000000000ULL /
				      (uint64_t)ovi.fps_num;

		if (f->sync_delay.load(std::memory_order_relaxed)) {
			/* take over the parent's audio sync offset once it's available */
			if (!f->sync_owned) {
				obs_source_t *parent =
					obs_filter_get_parent(f->context);
				if (parent) {
					f->saved_sync =
						obs_source_get_sync_offset(parent);
					f->sync_owned = true;
					f->applied_extra = -1;
				}
			}

			/* Commit an integer delay from the measured latency, but only let
			 * it move past a deadband + cooldown so the audio offset (and thus
			 * OBS's audio re-timing) holds steady instead of chasing jitter. */
			if (interval_ns && f->delay_ema_init) {
				const float target =
					(float)(f->delay_ema_ns / (double)interval_ns);
				if (f->commit_delay < 0) {
					f->commit_delay = (int)(target + 0.5f);
					f->last_commit_ns = now;
				} else if ((target > f->commit_delay + COMMIT_DEADBAND ||
					    target < f->commit_delay - COMMIT_DEADBAND) &&
					   now - f->last_commit_ns >= COMMIT_COOLDOWN_NS) {
					f->commit_delay = (int)(target + 0.5f);
					f->last_commit_ns = now;
				}
			}
			const int want = f->commit_delay < 0 ? 0 : f->commit_delay;

			ensure_delay_ring(f, w, h, want + 3);
			const int n = (int)f->ring.size();
			int applied_d = 0; /* frames the image is actually delayed by */
			if (n >= 2) {
				int d = want;
				if (d > n - 1)
					d = n - 1;
				gs_copy_texture(f->ring[f->ring_widx], full);
				if (d > 0 && f->ring_filled >= d) {
					const int idx =
						(f->ring_widx - d + n) % n;
					show = f->ring[idx];
					applied_d = d;
				}
				f->ring_widx = (f->ring_widx + 1) % n;
				if (f->ring_filled < n)
					f->ring_filled++;
			}

			/* Audio delay = the frames the image is actually delayed (0 during
			 * ring warm-up). Because `applied_d` is driven by the committed,
			 * deadbanded delay it only changes on a genuine, sustained latency
			 * shift -- so we re-apply (and OBS re-times) at most rarely. */
			const int64_t extra =
				interval_ns
					? (int64_t)applied_d * (int64_t)interval_ns
					: 0;
			if (f->sync_owned && extra != f->applied_extra) {
				obs_source_t *parent =
					obs_filter_get_parent(f->context);
				if (parent)
					obs_source_set_sync_offset(
						parent, f->saved_sync + extra);
				f->applied_extra = extra;
			}
		} else {
			release_audio_sync(f);
			if (!f->ring.empty())
				free_delay_ring(f);
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
		gs_effect_set_texture_srgb(f->p_image, show);
	else
		gs_effect_set_texture(f->p_image, show);
	gs_effect_set_texture(f->p_depthtex, f->depth_tex);
	gs_effect_set_float(f->p_strength, f->frac * f->disp_scale);
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
		gs_draw_sprite(show, 0, out_w, out_h);

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
