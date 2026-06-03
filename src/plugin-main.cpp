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
 * CMake (-DPLUGIN_VERSION, e.g. v0.2.3 for a tag build); the fallback below is
 * only used for IDE/standalone builds that don't define it. */
#ifndef REAL3D_VERSION
#define REAL3D_VERSION "0.2.3-dev"
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
/* Depth inference input dims (W x H), must match the exported ONNX. 16:9
 * (448x252, both multiples of 14) to match the source/per-eye aspect: the model
 * sees an undistorted image (vs an anamorphic square) and is ~0.73x the compute
 * of 392^2 (576 vs 784 tokens). The whole inference path is dimension-general,
 * so this plus the matching model file is the only change. */
static const int INFER_W = 448;
static const int INFER_H = 252;
/* After motion stops, keep inferring this many "settle" frames so the temporal
 * blend converges to its ghost-free steady state before the static-skip freezes
 * the depth (otherwise a motion trail / post-cut ghost gets frozen in). */
static const int SETTLE_FRAMES = 8;

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
/* A render gap longer than this means video_render was paused (hidden/disabled),
 * not just a slow frame. video_tick resets the latency state on inactivity for
 * the delay-owning case, but this render-side fallback also covers a sync-delay-
 * OFF hide (tick can't reach the parent then) and any graphics-thread stall. */
static const uint64_t RESUME_GAP_NS = 500000000ULL;

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
	/* n counts the terminating NUL (source length is -1). Size the buffer to n
	 * (room for the chars + NUL), convert, then trim the NUL from the length --
	 * sizing to n-1 and writing n would write the terminator slot. */
	int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
	if (n <= 0)
		return L"";
	std::wstring w((size_t)n, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], n);
	w.resize((size_t)n - 1);
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
	gs_texrender_t *rt_pre[2] = {nullptr, nullptr}; /* ping-pong halving pyramid */
	gs_texrender_t *rt_small = nullptr; /* downscaled to INFER_W*INFER_H */
	gs_stagesurf_t *stage[2] = {nullptr, nullptr}; /* GPU->CPU readback, ping-pong */
	int stage_cur = 0;             /* surface staged this tick; map the other */
	bool stage_primed = false;     /* false until both surfaces hold a frame */
	gs_texture_t *depth_tex = nullptr;  /* INFER_W*INFER_H R32F */

	/* tunables -- written by real3d_update (UI thread), read on the graphics
	 * thread (render / inference / size queries). Atomic (relaxed) since there
	 * is no inter-field ordering requirement, just race-free scalar access,
	 * matching sync_delay/ort.* below. */
	std::atomic<float> frac{0.018f}, convergence{0.5f}, swap_sign{1.0f};
	std::atomic<bool> full_sbs{true};
	std::atomic<int> sbs_size{0};   /* 0 = match source, 1 = 1080p (1920x1080/eye) */
	std::atomic<bool> eye_letterbox{false}; /* aspect-fit source into each eye (vs stretch) */
	std::atomic<bool> show_depth{false};
	std::atomic<bool> logged_dims{false};
	std::atomic<uint64_t> infer_interval_ns{66666666ULL}; /* depth cadence; 15 fps default */

	/* ONNX + flow-guided temporal stabiliser + worker */
	OrtDepth ort;
	FlowStabilizer flow;
	bool ort_ok = false;
	std::atomic<bool> ort_live{true}; /* false after a runtime Run() failure */
	std::thread worker;
	std::mutex m;
	std::condition_variable cv;
	std::vector<uint8_t> in_buf;   /* INFER_W*INFER_H * 4 RGBA, tight */
	std::vector<float> depth_buf;  /* INFER_W*INFER_H */
	bool input_ready = false, depth_ready = false, stop = false;
	uint32_t in_gen = 0;           /* (m) scene-cut generation of in_buf's frame */
	uint32_t out_gen = 0;          /* (m) generation depth_buf was computed for */
	uint64_t in_ts = 0;            /* (m) capture timestamp of in_buf's frame (ns) */
	uint64_t out_ts = 0;           /* (m) capture timestamp depth_buf was computed for */
	bool logged_first_depth = false;
	uint64_t last_submit_ns = 0;

	/* static-frame skip: when the downscaled input barely changes we reuse
	 * the depth already in depth_tex instead of re-running ONNX */
	std::vector<uint8_t> readback; /* reused INFER_W*INFER_H*4 readback scratch (graphics thread) */
	std::vector<uint8_t> prev_in; /* last *inferred* INFER_W*INFER_H*4 frame */
	std::atomic<bool> skip_static{true};       /* UI thread writes, graphics reads */
	std::atomic<float> static_thresh{1.0f};    /* mean abs RGB diff (0-255); 0 = exact */
	int settle_left = 0;          /* remaining settle inferences after motion */

	/* scene-cut handling (graphics thread only, except in_gen/out_gen under m).
	 * The disparity is flattened (2D) whenever the frame we are about to warp
	 * and the depth we currently hold are from different scenes -- i.e.
	 * shown_gen != depth_gen. Comparing the *displayed* frame's generation makes
	 * this correct in frame-matched delay mode too: there the shown frame is
	 * delayed by the ring, so the flat hold starts when the post-cut frame is
	 * actually displayed (not when the cut is detected) and lifts when the
	 * matching depth lands -- no needless flattening of the old scene's tail. */
	uint64_t last_detect_ns = 0;  /* cut-detection cadence (canvas fps, capped) */
	std::vector<uint8_t> prev_detect; /* previous frame sampled for cut detection */
	uint32_t scene_gen = 0;       /* ++ on each detected cut; tags freshly captured frames */
	uint32_t depth_gen = 0;       /* scene generation of the depth now in depth_tex */
	float disp_scale = 1.0f;      /* 0..1 multiplier on frac; ramps back after a cut */
	uint64_t stale_since_ns = 0;  /* when the current flat hold started (0 = not flat); failsafe */
	uint64_t last_render_ns = 0;  /* previous render timestamp, for fps-independent ramp */

	/* frame-matched delay mode */
	std::atomic<bool> sync_delay{false}; /* enable image-delay + audio sync */
	uint64_t cur_capture_ns = 0;  /* capture timestamp of this render's frame */
	uint64_t staged_ts[2] = {0, 0}; /* capture ts paired with each staging surface */
	double delay_ema_ns = 0.0;    /* measured pipeline latency (capture->depth), LPF */
	bool delay_ema_init = false;
	uint64_t latency_epoch_ns = 0; /* reject latency samples from captures before the last resume */
	int commit_delay = -1;        /* committed delay in frames (drives video+audio); -1 = unset */
	uint64_t last_commit_ns = 0;  /* throttles re-commits so audio offset stays steady */
	std::vector<gs_texture_t *> ring; /* recent full-res frames (delay line) */
	std::vector<uint32_t> ring_gen;   /* scene generation tag per ring slot */
	int ring_w = 0, ring_h = 0;   /* ring slot dims (rebuild on source resize) */
	int ring_widx = 0;            /* next slot to write */
	int ring_filled = 0;          /* slots written so far (for warm-up) */
	/* audio sync via the source's sync offset (OBS buffers the audio for us).
	 * All of this is touched ONLY on the graphics thread (render acquires/applies;
	 * tick/render/destroy restore), so no locking is needed. Detach is observed on
	 * the graphics thread via the parent's filter list (see real3d_video_tick) --
	 * we never touch sync state from the UI-thread filter_remove path, which is
	 * what made the previous atomic-flag handoff racy. */
	bool sync_owned = false;      /* we manage the parent's sync offset */
	obs_weak_source_t *sync_parent = nullptr; /* parent we took sync from; weak ref
		* kept so we can still restore after libobs detaches the filter (delete /
		* Undo restore_filters clears filter_parent before our destroy runs) */
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
	f->ring_gen.clear();
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
	f->ring_gen.assign((size_t)slots, 0);
	for (int i = 0; i < slots; ++i)
		f->ring[i] = gs_texture_create(w, h, GS_RGBA, 1, nullptr,
					       GS_RENDER_TARGET);
	f->ring_w = (int)w;
	f->ring_h = (int)h;
	f->ring_widx = 0;
	f->ring_filled = 0;
}

/* Restore the parent's audio sync offset we took over and reset the delay
 * pipeline's render-owned state. Idempotent. Graphics-thread only (render /
 * tick) or destroy (no concurrent render), so it's race-free. */
static void release_audio_sync(real3d_filter *f)
{
	/* Restore the offset through the retained weak ref -- never
	 * obs_filter_get_parent(), which is only valid inside render/filter_*
	 * callbacks, not in tick. */
	if (f->sync_owned && f->sync_parent) {
		obs_source_t *parent = obs_weak_source_get_source(f->sync_parent);
		if (parent) {
			obs_source_set_sync_offset(parent, f->saved_sync);
			obs_source_release(parent);
		}
	}
	f->sync_owned = false;
	if (f->sync_parent) {
		obs_weak_source_release(f->sync_parent);
		f->sync_parent = nullptr;
	}
	f->applied_extra = -1;
	f->commit_delay = -1;
	/* Reset the ring's warm-up state too: when this release is due to the filter
	 * going inactive (hidden/disabled via video_tick), the slots still hold
	 * pre-pause frames. Without this, video_render would immediately show a slot
	 * captured before the pause on resume (a few-frame rewind). Zeroing the
	 * fill/write index re-warms the ring (shows the live frame until it refills),
	 * matching the initial start-up behaviour. The textures are kept. */
	f->ring_widx = 0;
	f->ring_filled = 0;
}

static void worker_fn(real3d_filter *f)
{
	std::vector<uint8_t> local_in;
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
		const bool temporal = f->ort.temporal.load(std::memory_order_relaxed);
		const TemporalMode temporal_mode =
			(TemporalMode)f->ort.temporal_mode.load(std::memory_order_relaxed);

		/* raw ONNX depth -> normalise + flow-guided temporal stabilise */
		if (f->ort.Run(local_in.data(), raw)) {
			f->flow.process(
				local_in.data(),
				f->ort.width(), f->ort.height(), raw, stab,
				temporal, temporal_mode,
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
	const auto rel = std::memory_order_relaxed;
	f->frac.store(STRENGTH_FRAC[tier], rel); /* tier 0 -> 0 disparity (flat, A/B) */
	f->convergence.store((float)obs_data_get_double(s, "convergence"), rel);
	f->swap_sign.store(obs_data_get_bool(s, "swap") ? -1.0f : 1.0f, rel);
	bool prev = f->full_sbs.load(rel);
	bool full_sbs = obs_data_get_bool(s, "full_sbs");
	f->full_sbs.store(full_sbs, rel);
	f->sbs_size.store((int)obs_data_get_int(s, "sbs_size"), rel);
	f->eye_letterbox.store(obs_data_get_bool(s, "eye_letterbox"), rel);
	if (prev != full_sbs)
		f->logged_dims.store(false, rel); /* re-log new output size once */

	long long fps = obs_data_get_int(s, "infer_fps");
	if (fps < 1 || fps > 60)
		fps = 15;
	f->infer_interval_ns.store(1000000000ULL / (uint64_t)fps, rel);

	f->skip_static.store(obs_data_get_bool(s, "skip_static"), rel);
	f->static_thresh.store((float)obs_data_get_double(s, "static_thresh"), rel);

	f->ort.temporal.store(obs_data_get_bool(s, "temporal"),
			      std::memory_order_relaxed);
	/* The temporal mode selector was a dev-time A/B comparison knob; fixed to
	 * the most complete anti-trail mode (reactive mask + history clipping). The
	 * enum and the other code paths stay in flow_stabilizer.hpp so it's a
	 * one-line change to re-expose if ever needed. */
	f->ort.temporal_mode.store((int)TemporalMode::ReactiveClip,
				   std::memory_order_relaxed);
	f->ort.stab_strength.store((float)obs_data_get_double(s, "stabilize_strength"),
				   std::memory_order_relaxed);
	f->ort.depth_smooth.store((float)obs_data_get_double(s, "depth_smooth"),
				  std::memory_order_relaxed);
	f->show_depth.store(obs_data_get_bool(s, "show_depth"), rel);

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
	f->rt_pre[0] = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	f->rt_pre[1] = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	f->rt_small = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	f->stage[0] = gs_stagesurface_create(INFER_W, INFER_H, GS_RGBA);
	f->stage[1] = gs_stagesurface_create(INFER_W, INFER_H, GS_RGBA);
	std::vector<float> half((size_t)INFER_W * INFER_H, 0.5f);
	f->depth_tex = gs_texture_create(INFER_W, INFER_H, GS_R32F, 1,
					 nullptr, GS_DYNAMIC);
	gs_texture_set_image(f->depth_tex, (const uint8_t *)half.data(),
			     INFER_W * sizeof(float), false);
	obs_leave_graphics();

	if (model_path) {
		f->ort_ok = f->ort.Init(utf8_to_wide(model_path), INFER_W, INFER_H);
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
	if (f->rt_pre[0])
		gs_texrender_destroy(f->rt_pre[0]);
	if (f->rt_pre[1])
		gs_texrender_destroy(f->rt_pre[1]);
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

/* get_properties-time check: warn if a 'near Real 3D Deband' filter sits *below*
 * this one in the parent's filter chain. Filters apply top->bottom, so a deband
 * below us debands the warped stereo output instead of the source. enum_filters
 * yields filters in application order (top / source-side first), so a deband seen
 * *after* we have passed ourselves is mis-ordered. Re-evaluated whenever the
 * properties panel is (re)opened (there is no live reorder callback). */
struct deband_order_check {
	obs_source_t *self;
	bool seen_self;
	bool deband_below;
};

static void real3d_check_deband_order(obs_source_t *, obs_source_t *child, void *param)
{
	auto *oc = static_cast<deband_order_check *>(param);
	if (child == oc->self) {
		oc->seen_self = true;
		return;
	}
	const char *id = obs_source_get_id(child);
	if (oc->seen_self && id && strcmp(id, "near_real3d_deband") == 0)
		oc->deband_below = true;
}

static obs_properties_t *real3d_properties(void *data)
{
	auto *f = static_cast<real3d_filter *>(data);
	obs_properties_t *p = obs_properties_create();
	obs_property_t *q;

	/* ---- build identity (top of the panel) ---- */
	obs_properties_add_text(p, "build_info", REAL3D_BUILD_INFO,
				OBS_TEXT_INFO);

	/* Mis-ordered Deband warning. Only added when a deband filter is actually
	 * below us, so it never surfaces for users without the deband filter. */
	if (f) {
		deband_order_check oc{f->context, false, false};
		obs_source_t *parent = obs_filter_get_parent(f->context);
		if (parent)
			obs_source_enum_filters(parent, real3d_check_deband_order, &oc);
		if (oc.deband_below) {
			obs_property_t *w = obs_properties_add_text(
				p, "deband_order_warn",
				obs_module_text("debandorderwarn"), OBS_TEXT_INFO);
			obs_property_text_set_info_type(w, OBS_TEXT_INFO_WARNING);
		}
	}

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
	/* Parent/child settings use checkable groups: the toggle is the group's
	 * header checkbox and the dependent control sits (indented) inside it, so
	 * the relationship is visible and the child is disabled in place -- no
	 * jarring show/hide, and a disabled group can't be dragged. The checkbox
	 * state is stored under the group name, same key as the old bool. */
	obs_properties_t *g_skip = obs_properties_create();
	q = obs_properties_add_float_slider(g_skip, "static_thresh",
					    obs_module_text("staticthresh"), 0.0,
					    8.0, 0.1);
	obs_property_set_long_description(q, obs_module_text("staticthresh.desc"));
	q = obs_properties_add_group(gst, "skip_static",
				     obs_module_text("skipstatic"),
				     OBS_GROUP_CHECKABLE, g_skip);
	obs_property_set_long_description(q, obs_module_text("skipstatic.desc"));

	obs_properties_t *g_temp = obs_properties_create();
	q = obs_properties_add_float_slider(g_temp, "stabilize_strength",
					    obs_module_text("stabstrength"), 0.0,
					    1.0, 0.05);
	obs_property_set_long_description(q, obs_module_text("stabstrength.desc"));
	q = obs_properties_add_group(gst, "temporal", obs_module_text("temporal"),
				     OBS_GROUP_CHECKABLE, g_temp);
	obs_property_set_long_description(q, obs_module_text("temporal.desc"));

	q = obs_properties_add_float_slider(gst, "depth_smooth",
					    obs_module_text("edgesoft"), 0.0, 1.0,
					    0.05);
	obs_property_set_long_description(q, obs_module_text("edgesoft.desc"));
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
	obs_data_set_default_double(s, "stabilize_strength", 0.4);
	obs_data_set_default_double(s, "depth_smooth", 0.3);
	obs_data_set_default_bool(s, "sync_delay", false);
	obs_data_set_default_bool(s, "show_depth", false);
}

/* Per-eye target resolution: match the source, or a fixed preset (e.g. 1080p so
 * the SBS frame fits glasses with per-eye Full HD regardless of source size). */
static void eye_dims(const real3d_filter *f, uint32_t srcW, uint32_t srcH,
		     uint32_t &eyeW, uint32_t &eyeH)
{
	if (f->sbs_size.load(std::memory_order_relaxed) == 1) { eyeW = 1920; eyeH = 1080; }
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
	return f->full_sbs.load(std::memory_order_relaxed) ? eyeW * 2 : eyeW;
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

	/* Area-average downscale to the inference size (INFER_W x INFER_H) via a
	 * progressive 2:1 halving pyramid (a hand-rolled mipmap; libobs exposes no
	 * runtime mip-gen and texrender can't mip). A single steep bilinear reduction
	 * only reads a 2x2 footprint, so reducing by more than 2:1 (e.g. a 1920/4K
	 * source in one step) under-samples and lets aliasing + compression banding
	 * through. Halving repeatedly box-averages the whole footprint instead; the
	 * last step reduces to INFER_W x INFER_H, by which point each axis reduces by
	 * <=~2:1 so the 2x2 bilinear tap is sufficient. Inference path only -- the
	 * visible warp still samples the full-res frame, so sharpness is unaffected. */
	gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	gs_eparam_t *dimg = gs_effect_get_param_by_name(def, "image");
	struct vec4 clr;
	vec4_zero(&clr);
	const uint32_t PRE_W = INFER_W * 2, PRE_H = INFER_H * 2;

	gs_texture_t *src = full;
	uint32_t cw = gs_texture_get_width(full);
	uint32_t ch = gs_texture_get_height(full);
	int pp = 0; /* ping-pong index into rt_pre[] */
	while (cw > PRE_W || ch > PRE_H) {
		/* Halve each axis independently, only while it still exceeds 2x its
		 * inference target, so a wide/ultrawide source doesn't over-shrink its
		 * short axis before the final reduction. */
		const uint32_t nw = cw > PRE_W ? (cw + 1) / 2 : cw;
		const uint32_t nh = ch > PRE_H ? (ch + 1) / 2 : ch;
		gs_texrender_t *dst = f->rt_pre[pp];
		gs_texrender_reset(dst);
		if (!gs_texrender_begin(dst, nw, nh))
			return;
		gs_clear(GS_CLEAR_COLOR, &clr, 0.0f, 0);
		gs_ortho(0.0f, (float)nw, 0.0f, (float)nh, -100.0f, 100.0f);
		gs_effect_set_texture(dimg, src);
		while (gs_effect_loop(def, "Draw"))
			gs_draw_sprite(src, 0, nw, nh);
		gs_texrender_end(dst);
		src = gs_texrender_get_texture(dst);
		cw = nw;
		ch = nh;
		pp ^= 1; /* next pass writes the other buffer, reads this one */
	}

	gs_texrender_reset(f->rt_small);
	if (!gs_texrender_begin(f->rt_small, INFER_W, INFER_H))
		return;
	gs_clear(GS_CLEAR_COLOR, &clr, 0.0f, 0);
	gs_ortho(0.0f, (float)INFER_W, 0.0f, (float)INFER_H, -100.0f, 100.0f);
	gs_effect_set_texture(dimg, src);
	while (gs_effect_loop(def, "Draw"))
		gs_draw_sprite(src, 0, INFER_W, INFER_H);
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
	/* Reuse a member buffer instead of allocating each detect-tick: resize is a
	 * no-op once the capacity is established, so the per-frame readback fill no
	 * longer churns the heap on the graphics thread (this runs every tick, even
	 * while static-skipping). */
	f->readback.resize((size_t)INFER_W * INFER_H * 4);
	std::vector<uint8_t> &tight = f->readback;
	for (int y = 0; y < INFER_H; ++y)
		memcpy(&tight[(size_t)y * INFER_W * 4],
		       data + (size_t)y * linesize,
		       (size_t)INFER_W * 4);
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
		/* Just advance the generation and refresh the depth ASAP. The flat
		 * hold is derived at warp time from shown_gen != depth_gen, so it
		 * engages exactly when the post-cut frame reaches the screen (which,
		 * in delay mode, is several frames after this detection). */
		f->scene_gen++;
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
	if (now - f->last_submit_ns <
	    f->infer_interval_ns.load(std::memory_order_relaxed))
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
		f->skip_static.load(std::memory_order_relaxed) &&
		frame_is_static(tight, f->prev_in,
				f->static_thresh.load(std::memory_order_relaxed));
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
		/* Copy (not swap): tight aliases the reused f->readback, so swapping
		 * would hand its buffer to the worker and force a re-alloc next tick.
		 * The worker's swap returns an INFER-sized buffer to in_buf, so this
		 * copy reuses that capacity and stays alloc-free in the steady state. */
		f->in_buf = tight;
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

	/* Resume after a paused video_render (hide/disable, or a graphics stall):
	 * drop the delay pipeline's pre-pause state so a capture timestamp from
	 * before the gap isn't measured as `now - got_ts` (the whole pause) and fed
	 * to the latency EMA. video_tick handles this for the delay-owning case, but
	 * this also covers a sync-delay-OFF hide (the EMA still updates then, and tick
	 * can't reach the parent without ownership) and stalls tick can't see. */
	if (f->last_render_ns && now - f->last_render_ns > RESUME_GAP_NS) {
		f->stage_primed = false;
		f->latency_epoch_ns = now;
	}

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
					     INFER_W * sizeof(float), false);
			/* Record which scene this depth belongs to. The warp trusts the
			 * depth (lets the disparity ramp back) once the frame being shown
			 * is no newer than depth_gen; a stale pre-cut result is uploaded
			 * but the disparity stays flat until the matching depth lands. */
			f->depth_gen = got_gen;
			/* Measure the capture->depth pipeline latency (LPF). This is how
			 * far the image must be delayed in sync mode to match the depth.
			 * Only measure while sync delay is ON: the EMA is unused otherwise,
			 * and measuring with it off lets a hide/resume from any duration
			 * (which the tick reset and render-gap can't always cover when we
			 * don't own sync) poison the value for a later enable. */
			if (f->sync_delay.load(std::memory_order_relaxed) && got_ts &&
			    got_ts >= f->latency_epoch_ns && now > got_ts) {
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
				     INFER_W, INFER_H);
			}
		}
	}

	/* Frame-matched delay: pick which buffered frame to warp so it lines up with
	 * the depth we currently have, and delay the source audio to match. When the
	 * mode is off, release the audio sync and free the ring. shown_gen is the
	 * scene generation of whatever frame we end up warping (the delayed ring
	 * frame, or the current one when not delaying), used below to decide whether
	 * the depth we hold matches it. */
	gs_texture_t *show = full;
	uint32_t shown_gen = f->scene_gen;
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
					/* keep a weak ref so we can still restore after
					 * libobs detaches us (delete / Undo) */
					if (f->sync_parent)
						obs_weak_source_release(f->sync_parent);
					f->sync_parent =
						obs_source_get_weak_source(parent);
					f->applied_extra = -1;
					f->sync_owned = true;
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
				f->ring_gen[f->ring_widx] = f->scene_gen;
				if (d > 0 && f->ring_filled >= d) {
					const int idx =
						(f->ring_widx - d + n) % n;
					show = f->ring[idx];
					shown_gen = f->ring_gen[idx];
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

	/* Scene-cut disparity fade. Flatten to 2D whenever the frame we are about to
	 * warp (shown_gen) and the depth we hold (depth_gen) are from different
	 * scenes -- warping across a cut is the glitch we are avoiding. depth_tex
	 * holds only the latest depth, so a mismatch in *either* direction is unsafe:
	 *   shown_gen > depth_gen : the post-cut frame's depth hasn't arrived yet.
	 *   shown_gen < depth_gen : (delay mode) the forced post-cut inference landed
	 *     the next scene's depth while the ring is still showing the prior scene,
	 *     and that scene's matching depth has already been overwritten.
	 * Once the two line up, ramp back to full over ~1/DISP_RAMP_PER_SEC s so the
	 * 3D returns without a pop. A failsafe lifts the hold if they never converge
	 * (e.g. inference failure) so the filter can't get stuck flat. */
	{
		const bool depth_stale = shown_gen != f->depth_gen;
		if (depth_stale) {
			if (!f->stale_since_ns)
				f->stale_since_ns = now;
		} else {
			f->stale_since_ns = 0;
		}
		const bool failsafe =
			f->stale_since_ns &&
			now - f->stale_since_ns > FLATTEN_TIMEOUT_NS;
		const float dt = f->last_render_ns
					 ? (float)(now - f->last_render_ns) * 1e-9f
					 : 0.0f;
		f->last_render_ns = now;
		if (depth_stale && !failsafe) {
			f->disp_scale = 0.0f;
		} else if (f->disp_scale < 1.0f) {
			f->disp_scale += DISP_RAMP_PER_SEC * dt;
			if (f->disp_scale > 1.0f)
				f->disp_scale = 1.0f;
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
	const auto rel = std::memory_order_relaxed;
	const bool full_sbs = f->full_sbs.load(rel);
	gs_effect_set_texture(f->p_depthtex, f->depth_tex);
	gs_effect_set_float(f->p_strength, f->frac.load(rel) * f->disp_scale);
	gs_effect_set_float(f->p_conv, f->convergence.load(rel));
	gs_effect_set_float(f->p_swap, f->swap_sign.load(rel));
	gs_effect_set_float(f->p_usedepth,
			    (f->ort_ok && f->ort_live.load(rel)) ? 1.0f : 0.0f);
	gs_effect_set_float(f->p_showdepth, f->show_depth.load(rel) ? 1.0f : 0.0f);

	/* per-eye target size + optional aspect-fit (letterbox) of the source */
	uint32_t eyeW, eyeH;
	eye_dims(f, w, h, eyeW, eyeH);
	struct vec2 efit;
	efit.x = 1.0f;
	efit.y = 1.0f;
	if (f->eye_letterbox.load(rel) && w && h && eyeW && eyeH) {
		const float src_a = (float)w / (float)h;
		const float eye_a = (float)eyeW / (float)eyeH;
		const float ratio = src_a / eye_a;
		if (ratio >= 1.0f)
			efit.y = 1.0f / ratio; /* source wider -> bars top/bottom */
		else
			efit.x = ratio;        /* source taller -> bars left/right */
	}
	gs_effect_set_vec2(f->p_eyefit, &efit);

	const uint32_t out_w = full_sbs ? eyeW * 2 : eyeW;
	const uint32_t out_h = eyeH;
	if (!f->logged_dims.load(rel)) {
		f->logged_dims.store(true, rel);
		blog(LOG_INFO, "[near-real3d] output %ux%u (%s, source %ux%u) linear_srgb=%d",
		     out_w, out_h, full_sbs ? "Full-SBS" : "Half-SBS", w, h,
		     (int)linear_srgb);
	}
	while (gs_effect_loop(f->effect, "Draw"))
		gs_draw_sprite(show, 0, out_w, out_h);

	gs_enable_framebuffer_srgb(prev_fb);
}

/* obs_source_enum_filters callback: set found=true if our filter is in the list. */
static void find_self_in_filters(obs_source_t *, obs_source_t *child, void *param)
{
	auto **pair = static_cast<void **>(param); /* [0]=self, [1]=&found(bool) */
	if (child == pair[0])
		*static_cast<bool *>(pair[1]) = true;
}

/* Hand the parent's audio sync offset back when the delay mode stops being in
 * effect. video_render manages it while actively rendering, but stops being
 * called once the filter is disabled, hidden, or detached -- leaving the parent
 * with our audio delay (audio late while the now-undelayed video plays). tick
 * runs every frame for every source regardless of state (libobs ticks all
 * sources) and on the same graphics thread as render, so it's the single safe
 * place to detect "no longer active" and restore. Detach (delete AND Undo's
 * restore_filters, which skips filter_remove) is detected by checking we're
 * still in the parent's filter list -- a retained weak parent being visible
 * doesn't prove we're still attached to it. */
static void real3d_video_tick(void *data, float)
{
	auto *f = static_cast<real3d_filter *>(data);
	if (!f->sync_owned)
		return; /* nothing taken over -> nothing to restore */
	/* Resolve the retained weak ref (never obs_filter_get_parent() -- only valid
	 * inside render/filter_* callbacks, and it would race a UI-thread detach). */
	obs_source_t *parent =
		f->sync_parent ? obs_weak_source_get_source(f->sync_parent) : nullptr;
	bool attached = false;
	if (parent) {
		void *pair[2] = {f->context, &attached};
		obs_source_enum_filters(parent, find_self_in_filters, pair);
	}
	const bool active = f->sync_delay.load(std::memory_order_relaxed) &&
			    obs_source_enabled(f->context) && parent && attached &&
			    obs_source_showing(parent);
	if (parent)
		obs_source_release(parent);
	if (!active) {
		/* Drop the pre-pause latency state so a later resume (any duration --
		 * even a quick hide the render-side gap check would miss) can't feed the
		 * EMA a sample spanning the hidden period. */
		f->stage_primed = false;
		f->latency_epoch_ns = os_gettime_ns();
		release_audio_sync(f);
	}
}

/* ============================================================================
 * Standalone debanding filter (near_real3d_deband)
 *
 * The same mpv-style deband as the SBS warp's inline path, but as a plain 1:1
 * image filter so it can sit on any source / anywhere in a filter chain -- not
 * only on the 3D filter. Runs at the source resolution (taps + grain in true
 * source pixels; grain added at output resolution = best 8-bit dither). SRGB is
 * handled by libobs (OBS_SOURCE_SRGB) like obs-filters' sharpness_v2, so it
 * samples in linear light, matching the inline path. Note: chaining this before
 * the SBS filter still re-quantises at the SBS 8-bit capture, so for 3D-only use
 * the SBS filter's built-in Debanding group remains the no-loss path.
 * ============================================================================ */
struct deband_filter {
	obs_source_t *context = nullptr;
	gs_effect_t *effect = nullptr;
	gs_eparam_t *p_iters = nullptr, *p_thresh = nullptr, *p_range = nullptr,
		    *p_grain = nullptr, *p_image_size = nullptr;
	std::atomic<float> iters{1.0f}, threshold{12.0f}, range{4.0f},
		grain{9.0f};
};

static const char *deband_get_name(void *)
{
	return "near Real 3D Deband";
}

static void deband_update(void *data, obs_data_t *s)
{
	auto *f = static_cast<deband_filter *>(data);
	const auto rel = std::memory_order_relaxed;
	long long it = obs_data_get_int(s, "deband_iterations");
	if (it < 1 || it > 4)
		it = 1;
	f->iters.store((float)it, rel);
	f->threshold.store((float)obs_data_get_double(s, "deband_threshold"), rel);
	f->range.store((float)obs_data_get_double(s, "deband_range"), rel);
	f->grain.store((float)obs_data_get_double(s, "deband_grain"), rel);
}

static void *deband_create(obs_data_t *settings, obs_source_t *context)
{
	auto *f = new deband_filter();
	f->context = context;
	char *effect_path = obs_module_file("deband.effect");
	obs_enter_graphics();
	f->effect = gs_effect_create_from_file(effect_path, nullptr);
	if (f->effect) {
		f->p_iters = gs_effect_get_param_by_name(f->effect, "deband_iters");
		f->p_thresh = gs_effect_get_param_by_name(f->effect, "deband_threshold");
		f->p_range = gs_effect_get_param_by_name(f->effect, "deband_range");
		f->p_grain = gs_effect_get_param_by_name(f->effect, "deband_grain");
		f->p_image_size = gs_effect_get_param_by_name(f->effect, "image_size");
	}
	obs_leave_graphics();
	bfree(effect_path);
	if (!f->effect) {
		blog(LOG_ERROR, "[near-real3d] deband.effect missing -> filter disabled");
		delete f;
		return nullptr;
	}
	deband_update(f, settings);
	return f;
}

static void deband_destroy(void *data)
{
	auto *f = static_cast<deband_filter *>(data);
	obs_enter_graphics();
	if (f->effect)
		gs_effect_destroy(f->effect);
	obs_leave_graphics();
	delete f;
}

static obs_properties_t *deband_properties(void *)
{
	obs_properties_t *p = obs_properties_create();
	obs_properties_add_text(p, "build_info", REAL3D_BUILD_INFO, OBS_TEXT_INFO);
	obs_property_t *it = obs_properties_add_list(
		p, "deband_iterations", obs_module_text("debanditer"),
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(it, obs_module_text("debanditer.1"), 1);
	obs_property_list_add_int(it, obs_module_text("debanditer.2"), 2);
	obs_property_list_add_int(it, obs_module_text("debanditer.3"), 3);
	obs_property_list_add_int(it, obs_module_text("debanditer.4"), 4);
	obs_property_set_long_description(it, obs_module_text("debanditer.desc"));
	obs_property_t *q = obs_properties_add_float_slider(
		p, "deband_threshold", obs_module_text("debandthresh"), 0.0,
		200.0, 1.0);
	obs_property_set_long_description(q, obs_module_text("debandthresh.desc"));
	q = obs_properties_add_float_slider(p, "deband_range",
					    obs_module_text("debandrange"), 1.0,
					    64.0, 1.0);
	obs_property_set_long_description(q, obs_module_text("debandrange.desc"));
	q = obs_properties_add_float_slider(p, "deband_grain",
					    obs_module_text("debandgrain"), 0.0,
					    100.0, 1.0);
	obs_property_set_long_description(q, obs_module_text("debandgrain.desc"));
	return p;
}

static void deband_defaults(obs_data_t *s)
{
	obs_data_set_default_int(s, "deband_iterations", 1);
	obs_data_set_default_double(s, "deband_threshold", 12.0);
	obs_data_set_default_double(s, "deband_range", 4.0);
	obs_data_set_default_double(s, "deband_grain", 9.0);
}

static void deband_render(void *data, gs_effect_t *)
{
	auto *f = static_cast<deband_filter *>(data);
	obs_source_t *target = obs_filter_get_target(f->context);
	if (!f->effect || !target) {
		obs_source_skip_video_filter(f->context);
		return;
	}

	/* SRGB-aware like obs-filters' sharpness_v2: SDR passes through as
	 * GS_RGBA / GS_CS_SRGB (sampled in linear light); HDR (extended) sources
	 * are passed through untouched. */
	const enum gs_color_space pref[] = {GS_CS_SRGB, GS_CS_SRGB_16F,
					    GS_CS_709_EXTENDED};
	const enum gs_color_space space =
		obs_source_get_color_space(target, 3, pref);
	if (space == GS_CS_709_EXTENDED) {
		obs_source_skip_video_filter(f->context);
		return;
	}
	const enum gs_color_format fmt = gs_get_format_from_space(space);
	if (!obs_source_process_filter_begin_with_color_space(
		    f->context, fmt, space, OBS_ALLOW_DIRECT_RENDERING))
		return;

	const auto rel = std::memory_order_relaxed;
	gs_effect_set_float(f->p_iters, f->iters.load(rel));
	gs_effect_set_float(f->p_thresh, f->threshold.load(rel));
	gs_effect_set_float(f->p_range, f->range.load(rel));
	gs_effect_set_float(f->p_grain, f->grain.load(rel));
	struct vec2 imsz;
	imsz.x = (float)obs_source_get_width(target);
	imsz.y = (float)obs_source_get_height(target);
	if (imsz.x < 1.0f)
		imsz.x = 1.0f;
	if (imsz.y < 1.0f)
		imsz.y = 1.0f;
	gs_effect_set_vec2(f->p_image_size, &imsz);

	/* Premultiplied-alpha draw, like every obs-filters video filter, so the
	 * result composites correctly on sources with alpha (no edge darkening). */
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);
	obs_source_process_filter_end(f->context, f->effect, 0, 0);
	gs_blend_state_pop();
}

static enum gs_color_space deband_get_color_space(void *data, size_t,
						  const enum gs_color_space *)
{
	auto *f = static_cast<deband_filter *>(data);
	const enum gs_color_space pref[] = {GS_CS_SRGB, GS_CS_SRGB_16F,
					    GS_CS_709_EXTENDED};
	return obs_source_get_color_space(obs_filter_get_target(f->context), 3,
					  pref);
}

static struct obs_source_info real3d_filter_info = {};
static struct obs_source_info real3d_deband_info = {};

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
	real3d_filter_info.video_tick = real3d_video_tick;
	obs_register_source(&real3d_filter_info);

	/* Standalone deband filter (shares this module). SRGB-aware; no async/tick. */
	real3d_deband_info.id = "near_real3d_deband";
	real3d_deband_info.type = OBS_SOURCE_TYPE_FILTER;
	real3d_deband_info.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB;
	real3d_deband_info.get_name = deband_get_name;
	real3d_deband_info.create = deband_create;
	real3d_deband_info.destroy = deband_destroy;
	real3d_deband_info.update = deband_update;
	real3d_deband_info.get_properties = deband_properties;
	real3d_deband_info.get_defaults = deband_defaults;
	real3d_deband_info.video_render = deband_render;
	real3d_deband_info.video_get_color_space = deband_get_color_space;
	obs_register_source(&real3d_deband_info);

	blog(LOG_INFO, "[near-real3d] loaded: %s (libobs %d.%d.%d)",
	     REAL3D_BUILD_INFO, LIBOBS_API_MAJOR_VER, LIBOBS_API_MINOR_VER,
	     LIBOBS_API_PATCH_VER);
	return true;
}

void obs_module_unload(void)
{
	blog(LOG_INFO, "[near-real3d] unloaded");
}
