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
 * CMake (-DPLUGIN_VERSION, e.g. v0.3.1 for a tag build); the fallback below is
 * only used for IDE/standalone builds that don't define it. */
#ifndef REAL3D_VERSION
#define REAL3D_VERSION "0.0.0-nocmake" /* sentinel: CMake always passes the real one */
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
/* Default depth inference input dims (W x H) -- the fallback used when no ONNX
 * model loads (luminance-depth path). When a model loads, the *actual* dims are
 * read from its fixed input shape (OrtDepth::Init) into the per-filter
 * infer_w/infer_h, so shipping a lighter export (e.g. 392x224 -lite, ~1.4x the
 * speed on a weak iGPU at near-full depth quality) drives the whole pipeline with
 * no code change. 448x252 is 16:9 (both multiples of 14) to match the source/per-
 * eye aspect; the inference path is fully dimension-general. */
static const int DEFAULT_INFER_W = 448;
static const int DEFAULT_INFER_H = 252;
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
/* Auto-suppression eases the disparity toward the scene confidence; slower than
 * the post-cut ramp so the 3D breathes gently in/out (~0.5 s end to end) instead
 * of snapping when a low-confidence stretch starts or ends. */
static const float CONF_RAMP_PER_SEC = 2.0f;
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
 * warp applies slightly-stale depth to the current frame. In this mode we delay
 * the displayed image by that latency (a ring of recent full-res frames) and,
 * instead of always warping with the *latest* depth, we keep a small cache of
 * recent depths and pick the one that corresponds to the frame actually being
 * shown -- matched by scene generation (hard boundary, never mix scenes) then by
 * capture timestamp (which frame within the scene). This keeps the 3D alive
 * across a hard cut even when the off-cadence post-cut inference returns early,
 * because the prior scene's depth is still cached for the prior-scene frames the
 * ring is still showing. Cost: the displayed video is delayed by the latency
 * (~0.1 s) plus a little extra VRAM. Audio is delayed to match via the source's
 * sync offset. */
static const int DELAY_RING_MAX = 16;            /* cap on buffered frames (VRAM bound) */
/* Depth cache: enough recent depths to cover the delay window (sized per frame
 * from the committed delay and the inference cadence). R32F infer_w x infer_h is
 * ~0.43 MB/slot, so even the cap is a few MB. */
static const int DEPTH_CACHE_MAX = 20;           /* hard cap on cached depths (VRAM bound) */
static const int DEPTH_CACHE_MARGIN = 2;         /* spare slots beyond the delay window */
static const float DELAY_EMA = 0.85f;            /* latency low-pass (per inference) */
/* The committed delay (frames) drives BOTH the video ring and the audio sync
 * offset. Video can change cheaply every frame, but changing the audio offset
 * makes OBS re-time the audio (it grows OBS's one-way "dynamically increasing"
 * audio buffer, which never shrinks in-session), so we only re-commit when the
 * measured latency leaves a deadband around the current value AND a cooldown has
 * passed -- keeping the offset rock-steady in the steady state. The deadband is
 * wider than 1 frame on purpose: when the true latency sits near a half-frame
 * boundary (e.g. ~2.5f at 24 fps) a narrower band lets ordinary compute-time
 * jitter limit-cycle the commit up/down each cooldown, and every up-swing
 * ratchets the audio buffer. >1 frame holds the commit put; the margin + depth
 * cache absorb the resulting <=1 frame of latency mismatch. */
static const float COMMIT_DEADBAND = 1.5f;       /* frames: ignore up to ~1-frame jitter */
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

/* One cached depth result. The cache (graphics thread only) holds a few recent
 * depths so frame-matched delay mode can warp the frame being *shown* with the
 * depth that corresponds to it, not just the latest one. `gen` is the scene-cut
 * generation it was computed for (hard match), `ts` the capture timestamp of its
 * source frame (fine match within a scene). `valid` is false until first written. */
struct depth_slot {
	gs_texture_t *tex = nullptr;  /* infer_w*infer_h R32F */
	uint32_t gen = 0;
	uint64_t ts = 0;
	bool valid = false;
};

struct real3d_filter {
	obs_source_t *context = nullptr;
	gs_effect_t *effect = nullptr;
	gs_eparam_t *p_image = nullptr, *p_strength = nullptr,
		    *p_conv = nullptr, *p_grading = nullptr,
		    *p_silhouette = nullptr, *p_swap = nullptr,
		    *p_usedepth = nullptr, *p_depthtex = nullptr,
		    *p_showdepth = nullptr, *p_depthtexel = nullptr,
		    *p_eyefit = nullptr,
		    *p_dither = nullptr, *p_outsize = nullptr,
		    *p_debug = nullptr;

	gs_texrender_t *rt_full = nullptr;  /* captured input at WxH */
	gs_texrender_t *rt_pre[2] = {nullptr, nullptr}; /* ping-pong halving pyramid */
	gs_texrender_t *rt_small = nullptr; /* downscaled to infer_w*infer_h */
	gs_stagesurf_t *stage[2] = {nullptr, nullptr}; /* GPU->CPU readback, ping-pong */
	int stage_cur = 0;             /* surface staged this tick; map the other */
	bool stage_primed = false;     /* false until both surfaces hold a frame */
	/* Depth cache: recent depths keyed by (gen, ts) so the warp can pick the one
	 * matching the frame being shown (see frame-matched delay mode). depth_flat is
	 * a constant flat-0.5 fallback bound when no cached depth matches (warm-up /
	 * unmatched), where the disparity is held at 0 anyway so its content is moot.
	 * Graphics thread only. */
	std::vector<depth_slot> depth_cache;
	int depth_widx = 0;            /* next cache slot to overwrite (round-robin) */
	int depth_used = 0;            /* effective live slots (<= allocated); 1 in debug A/B */
	gs_texture_t *depth_flat = nullptr; /* infer_w*infer_h R32F, flat 0.5 fallback */

	/* tunables -- written by real3d_update (UI thread), read on the graphics
	 * thread (render / inference / size queries). Atomic (relaxed) since there
	 * is no inter-field ordering requirement, just race-free scalar access,
	 * matching sync_delay/ort.* below. */
	std::atomic<float> frac{0.018f}, convergence{0.5f}, swap_sign{1.0f};
	std::atomic<float> grading{0.0f}; /* 0 = linear disparity; 1 = full tanh S-curve */
	std::atomic<float> silhouette{0.35f}; /* directional disparity clamp at depth edges */
	std::atomic<bool> full_sbs{true};
	std::atomic<int> sbs_size{0};   /* 0 = match source, 1 = 1080p (1920x1080/eye) */
	std::atomic<bool> eye_letterbox{false}; /* aspect-fit source into each eye (vs stretch) */
	std::atomic<float> dither{12.0f}; /* output dither amount (mpv-grain scale); 0 = off */
	/* auto 3D-suppression: ease the disparity down in low-confidence scenes
	 * (sky / starfield / fog / flat UI / strong camera shake) where the depth is
	 * fabricated/unstable -- "don't thrash" beats "guess". The confidence comes
	 * from the stabiliser's mean depth residual, so this only acts with temporal
	 * stabilisation on. */
	std::atomic<bool> auto_suppress{true};
	std::atomic<bool> show_depth{false};
	/* debug: tint the warp by per-frame depth-match state (red = 2D fallback,
	 * yellow = ramping) so the otherwise-invisible cut behaviour is visible; and
	 * force the depth cache to a single live slot, reproducing the pre-cache
	 * "latest depth only" behaviour for an instant A/B of the matching fix. */
	std::atomic<bool> debug_overlay{false};
	std::atomic<bool> debug_cache1{false};
	std::atomic<bool> logged_dims{false};
	std::atomic<uint64_t> infer_interval_ns{66666666ULL}; /* depth cadence; 15 fps default */

	/* ONNX + flow-guided temporal stabiliser + worker */
	OrtDepth ort;
	FlowStabilizer flow;
	bool ort_ok = false;
	/* Effective inference dims for this filter: read from the loaded model's
	 * input shape (OrtDepth::Init), or the default when no model loaded. Set once
	 * in real3d_create before the worker starts, then read-only (graphics thread
	 * + worker), so no atomic is needed. All GPU resources/buffers below size to
	 * these. */
	int infer_w = DEFAULT_INFER_W, infer_h = DEFAULT_INFER_H;
	std::atomic<bool> ort_live{true}; /* false after a runtime Run() failure */
	std::thread worker;
	std::mutex m;
	std::condition_variable cv;
	std::vector<uint8_t> in_buf;   /* infer_w*infer_h * 4 RGBA, tight */
	std::vector<float> depth_buf;  /* infer_w*infer_h */
	bool input_ready = false, depth_ready = false, stop = false;
	uint32_t in_gen = 0;           /* (m) scene-cut generation of in_buf's frame */
	bool in_forced = false;        /* (m) off-cadence cut-forced submission */
	uint32_t out_gen = 0;          /* (m) generation depth_buf was computed for */
	bool out_forced = false;       /* (m) depth_buf came from a cut-forced submission */
	float out_conf = 1.0f;         /* (m) scene confidence for depth_buf (auto 3D-suppress) */
	uint64_t in_ts = 0;            /* (m) capture timestamp of in_buf's frame (ns) */
	uint64_t out_ts = 0;           /* (m) capture timestamp depth_buf was computed for */
	/* debug latency decomposition: submit time of in_buf, and the worker's
	 * submit->pickup wait + compute time for the produced depth. Lets the log
	 * split the capture->depth latency so we can see where the cut-path excess
	 * over `commit` actually goes (worker busy-wait vs inference compute). */
	uint64_t in_submit_ns = 0;     /* (m) os_gettime_ns when in_buf was queued */
	uint64_t out_wait_ns = 0;      /* (m) submit -> worker started Run */
	uint64_t out_compute_ns = 0;   /* (m) Run + temporal stabilise */
	bool logged_first_depth = false;
	uint32_t last_logged_depth_gen = 0xFFFFFFFFu; /* debug: log first depth per gen once */
	uint64_t last_submit_ns = 0;

	/* static-frame skip: when the downscaled input barely changes we reuse
	 * the depth already cached instead of re-running ONNX */
	std::vector<uint8_t> readback; /* reused infer_w*infer_h*4 readback scratch (graphics thread) */
	std::vector<uint8_t> prev_in; /* last *inferred* infer_w*infer_h*4 frame */
	std::atomic<bool> skip_static{true};       /* UI thread writes, graphics reads */
	std::atomic<float> static_thresh{1.0f};    /* mean abs RGB diff (0-255); 0 = exact */
	int settle_left = 0;          /* remaining settle inferences after motion */

	/* scene-cut handling (graphics thread only, except in_gen/out_gen under m).
	 * The disparity is flattened (2D) whenever the frame we are about to warp has
	 * no matching cached depth (same scene generation, near capture timestamp).
	 * Matching against the *displayed* frame makes this correct in frame-matched
	 * delay mode too: there the shown frame is delayed by the ring, so the flat
	 * hold engages only when an unmatched post-cut frame actually reaches the
	 * screen and lifts the moment its matching depth lands -- the old scene's tail
	 * keeps its 3D from the cached prior-scene depth. */
	uint64_t last_detect_ns = 0;  /* cut-detection cadence (canvas fps, capped) */
	std::vector<uint8_t> prev_detect; /* previous frame sampled for cut detection */
	uint32_t scene_gen = 0;       /* ++ on each detected cut; tags freshly captured frames */
	float disp_scale = 1.0f;      /* 0..1 multiplier on frac; ramps back after a cut */
	/* auto 3D-suppression: scene_conf is the latest confidence handed off by the
	 * worker (1 = full 3D, down to FlowStabilizer's floor); conf_scale eases
	 * toward it each render so the disparity glides rather than steps. */
	float scene_conf = 1.0f;
	float conf_scale = 1.0f;
	uint64_t stale_since_ns = 0;  /* when the current flat hold started (0 = not flat); failsafe */
	bool stale_logged = false;    /* debug: failsafe-lift warning logged once per hold */
	uint64_t depth_match_dt = 0;  /* debug: |shown_ts - matched depth ts| from the last pick */
	uint64_t last_render_ns = 0;  /* previous render timestamp, for fps-independent ramp */

	/* frame-matched delay mode */
	std::atomic<bool> sync_delay{false}; /* enable image-delay + audio sync */
	/* Extra ring delay beyond the measured latency (frames). The post-cut depth
	 * lands ~1 frame after the new-scene frames would reach the screen when the
	 * delay equals the latency exactly, so the new scene's leading frames briefly
	 * have no matching depth (a short 2D flash). A small margin makes the depth
	 * win that race so a cut stays 3D, at the cost of a touch more latency. */
	std::atomic<int> delay_margin{2};
	uint64_t cur_capture_ns = 0;  /* capture timestamp of this render's frame */
	uint64_t staged_ts[2] = {0, 0}; /* capture ts paired with each staging surface */
	double delay_ema_ns = 0.0;    /* measured pipeline latency (capture->depth), LPF */
	bool delay_ema_init = false;
	uint64_t latency_epoch_ns = 0; /* reject latency samples from captures before the last resume */
	int commit_delay = -1;        /* committed delay in frames (drives video+audio); -1 = unset */
	int applied_delay = 0;        /* frames the image was actually delayed last render (debug log) */
	uint64_t last_commit_ns = 0;  /* throttles re-commits so audio offset stays steady */
	std::vector<gs_texture_t *> ring; /* recent full-res frames (delay line) */
	std::vector<uint32_t> ring_gen;   /* scene generation tag per ring slot */
	std::vector<uint64_t> ring_ts;    /* capture timestamp per ring slot (ns); 0 = unfilled */
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
	/* Anti-compounding: OBS persists the parent's sync_offset, so the `extra` we
	 * add leaks into the saved value and (re-read as a baseline next session)
	 * would compound. sync_leak_ns is how much of the parent's *current* offset is
	 * our addition; the filter's save callback persists it, and on the next load
	 * we subtract it back out (load_leak_ns, applied once) to recover the user's
	 * true baseline. atomic: written on the graphics thread, read by save (UI). */
	std::atomic<int64_t> sync_leak_ns{0};
	int64_t load_leak_ns = 0;     /* our leak persisted last session (to subtract) */
	bool load_leak_pending = true; /* do the one-time on-load correction */
};

/* Destroy the delay ring. Caller must hold the graphics context. */
static void free_delay_ring(real3d_filter *f)
{
	for (gs_texture_t *t : f->ring)
		if (t)
			gs_texture_destroy(t);
	f->ring.clear();
	f->ring_gen.clear();
	f->ring_ts.clear();
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
	f->ring_ts.assign((size_t)slots, 0);
	for (int i = 0; i < slots; ++i)
		f->ring[i] = gs_texture_create(w, h, GS_RGBA, 1, nullptr,
					       GS_RENDER_TARGET);
	f->ring_w = (int)w;
	f->ring_h = (int)h;
	f->ring_widx = 0;
	f->ring_filled = 0;
}

/* Destroy the depth cache. Caller must hold the graphics context. */
static void free_depth_cache(real3d_filter *f)
{
	for (depth_slot &s : f->depth_cache)
		if (s.tex)
			gs_texture_destroy(s.tex);
	f->depth_cache.clear();
	f->depth_widx = 0;
}

/* Grow the depth cache to `slots` R32F textures (infer_w x infer_h). Append-only:
 * existing slots (and their cached gen/ts/valid) are kept so a grow doesn't drop
 * depths mid-stream; dims never change so it never rebuilds. Caller holds the
 * graphics context. */
static void ensure_depth_cache(real3d_filter *f, int slots)
{
	if (slots < 2)
		slots = 2;
	if (slots > DEPTH_CACHE_MAX)
		slots = DEPTH_CACHE_MAX;
	const int have = (int)f->depth_cache.size();
	if (have >= slots)
		return;
	f->depth_cache.resize((size_t)slots);
	for (int i = have; i < slots; ++i) {
		depth_slot &s = f->depth_cache[i];
		s.tex = gs_texture_create(f->infer_w, f->infer_h, GS_R32F, 1,
					  nullptr, GS_DYNAMIC);
		s.gen = 0;
		s.ts = 0;
		s.valid = false;
	}
}

/* Effective delay window in frames = measured latency (commit_delay) + the user
 * margin, or 0 until a latency is committed. This single value drives BOTH the
 * video ring delay and the depth-cache sizing, which must agree -- if the cache
 * covered fewer frames than the ring delays, the matching depth would be evicted
 * before its frame is shown (a spurious 2D). Keep the formula here only. */
static int wanted_delay_frames(const real3d_filter *f)
{
	if (f->commit_delay < 0)
		return 0;
	return f->commit_delay + f->delay_margin.load(std::memory_order_relaxed);
}

/* How many depth slots are needed to keep the depth matching the oldest shown
 * (delayed) frame from being evicted before it is displayed. While that frame is
 * in the ring (the delay window), new depths keep arriving and overwriting cache
 * slots; production is capped at one per canvas frame, so over the window at most
 * `delay_frames * interval / max(infer_interval, interval)` depths land. Plus a
 * margin, clamped to the cap. Minimum 3 so non-delay mode keeps the latest depth
 * plus a little history for timestamp matching. */
static int wanted_depth_slots(const real3d_filter *f, uint64_t interval_ns)
{
	int delay_frames = wanted_delay_frames(f);
	if (delay_frames > DELAY_RING_MAX - 1)
		delay_frames = DELAY_RING_MAX - 1; /* effective delay is ring-bound */
	int slots = 3;
	const uint64_t infer_ns =
		f->infer_interval_ns.load(std::memory_order_relaxed);
	if (interval_ns && infer_ns && delay_frames > 0) {
		const uint64_t eff = infer_ns > interval_ns ? infer_ns : interval_ns;
		const double depths =
			(double)delay_frames * (double)interval_ns / (double)eff;
		const int need = (int)(depths + 0.999) + DEPTH_CACHE_MARGIN;
		if (need > slots)
			slots = need;
	}
	if (slots > DEPTH_CACHE_MAX)
		slots = DEPTH_CACHE_MAX;
	return slots;
}

/* Pick the cached depth for the frame being shown: scene generation must match
 * exactly (never warp across a cut), then the nearest capture timestamp within
 * that scene. Returns the flat fallback (and sets found=false) when nothing
 * matches -- the caller then holds the disparity flat (2D). Graphics thread. */
static gs_texture_t *pick_depth(real3d_filter *f, uint32_t shown_gen,
				uint64_t shown_ts, bool &found)
{
	int best = -1;
	uint64_t best_dt = 0;
	for (int i = 0; i < f->depth_used; ++i) {
		const depth_slot &s = f->depth_cache[i];
		if (!s.valid || s.gen != shown_gen)
			continue;
		const uint64_t dt = shown_ts > s.ts ? shown_ts - s.ts
						    : s.ts - shown_ts;
		if (best < 0 || dt < best_dt) {
			best = i;
			best_dt = dt;
		}
	}
	found = best >= 0;
	f->depth_match_dt = found ? best_dt : 0; /* debug: how close the match was (ns) */
	return found ? f->depth_cache[best].tex : f->depth_flat;
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
			if (f->debug_overlay.load(std::memory_order_relaxed))
				blog(LOG_INFO,
				     "[near-real3d] audio sync released: restored parent "
				     "offset to %.1f ms",
				     (double)f->saved_sync / 1e6);
			obs_source_release(parent);
		}
	}
	f->sync_owned = false;
	/* Our delay no longer rides the parent's offset, so nothing of ours is left in
	 * the persisted value (we restored the true baseline above). */
	f->sync_leak_ns.store(0, std::memory_order_relaxed);
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
		uint64_t submit = 0;
		bool forced = false;
		{
			std::unique_lock<std::mutex> lk(f->m);
			f->cv.wait(lk, [&] { return f->input_ready || f->stop; });
			if (f->stop)
				return;
			local_in.swap(f->in_buf);
			gen = f->in_gen; /* which cut generation this frame belongs to */
			ts = f->in_ts;   /* capture timestamp, for the delay-mode latency measure */
			submit = f->in_submit_ns; /* when it was queued, for the wait measure */
			forced = f->in_forced; /* cut-forced -> excluded from the latency EMA */
			f->input_ready = false;
		}
		const bool temporal = f->ort.temporal.load(std::memory_order_relaxed);
		const TemporalMode temporal_mode =
			(TemporalMode)f->ort.temporal_mode.load(std::memory_order_relaxed);

		/* raw ONNX depth -> normalise + flow-guided temporal stabilise */
		const uint64_t run_start = os_gettime_ns();
		if (f->ort.Run(local_in.data(), raw)) {
			f->flow.process(
				local_in.data(),
				f->ort.width(), f->ort.height(), raw, stab,
				temporal, temporal_mode,
				f->ort.stab_strength.load(std::memory_order_relaxed),
				f->ort.depth_smooth.load(std::memory_order_relaxed) * 6.0f);
			const uint64_t done = os_gettime_ns();
			{
				std::lock_guard<std::mutex> lk(f->m);
				f->depth_buf.swap(stab);
				f->out_gen = gen; /* tag the result with its cut generation */
				f->out_conf = f->flow.confidence(); /* for auto 3D-suppress */
				f->out_forced = forced; /* propagate the cut-forced tag */
				f->out_ts = ts;   /* ...and the frame's capture timestamp */
				/* latency decomposition (debug): how long it sat queued before
				 * the worker picked it up, and how long the compute itself took. */
				f->out_wait_ns =
					run_start > submit ? run_start - submit : 0;
				f->out_compute_ns = done - run_start;
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
	f->grading.store((float)obs_data_get_double(s, "grading"), rel);
	f->silhouette.store((float)obs_data_get_double(s, "silhouette"), rel);
	f->swap_sign.store(obs_data_get_bool(s, "swap") ? -1.0f : 1.0f, rel);
	bool prev = f->full_sbs.load(rel);
	bool full_sbs = obs_data_get_bool(s, "full_sbs");
	f->full_sbs.store(full_sbs, rel);
	f->sbs_size.store((int)obs_data_get_int(s, "sbs_size"), rel);
	f->eye_letterbox.store(obs_data_get_bool(s, "eye_letterbox"), rel);
	f->dither.store((float)obs_data_get_double(s, "dither"), rel);
	f->auto_suppress.store(obs_data_get_bool(s, "auto_suppress"), rel);
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
	f->debug_overlay.store(obs_data_get_bool(s, "debug_overlay"), rel);
	f->debug_cache1.store(obs_data_get_bool(s, "debug_cache1"), rel);

	/* Frame-matched delay: just record intent here; acquiring/releasing the
	 * parent's audio sync offset and (re)building the frame ring happen on the
	 * graphics thread in real3d_video_render, where the parent is always valid. */
	f->sync_delay.store(obs_data_get_bool(s, "sync_delay"),
			    std::memory_order_relaxed);
	long long margin = obs_data_get_int(s, "delay_margin");
	if (margin < 0)
		margin = 0;
	if (margin > 4)
		margin = 4;
	f->delay_margin.store((int)margin, std::memory_order_relaxed);
}

static void *real3d_create(obs_data_t *settings, obs_source_t *context)
{
	auto *f = new real3d_filter();
	f->context = context;

	char *effect_path = obs_module_file("near-real3d.effect");
	char *model_path = obs_module_file("depth_anything_v2_small.onnx");

	/* Load the depth model first: its fixed input shape determines the inference
	 * dims (infer_w/infer_h) that the GPU resources below are sized to. On failure
	 * we fall back to luminance depth at the default dims. */
	if (model_path) {
		f->ort_ok = f->ort.Init(utf8_to_wide(model_path));
		if (!f->ort_ok)
			blog(LOG_WARNING, "[near-real3d] ONNX init failed (%s) "
					  "-> luminance-depth fallback",
			     f->ort.last_error().c_str());
	} else {
		blog(LOG_WARNING, "[near-real3d] model not found -> luminance-depth fallback");
	}
	if (f->ort_ok) {
		f->infer_w = f->ort.width();
		f->infer_h = f->ort.height();
		blog(LOG_INFO, "[near-real3d] depth model loaded (%dx%d)",
		     f->infer_w, f->infer_h);
	}

	obs_enter_graphics();
	f->effect = gs_effect_create_from_file(effect_path, nullptr);
	if (f->effect) {
		f->p_image = gs_effect_get_param_by_name(f->effect, "image");
		f->p_strength = gs_effect_get_param_by_name(f->effect, "strength");
		f->p_conv = gs_effect_get_param_by_name(f->effect, "convergence");
		f->p_grading = gs_effect_get_param_by_name(f->effect, "grading");
		f->p_silhouette = gs_effect_get_param_by_name(f->effect, "silhouette");
		f->p_swap = gs_effect_get_param_by_name(f->effect, "swap_sign");
		f->p_usedepth = gs_effect_get_param_by_name(f->effect, "use_depth_tex");
		f->p_depthtex = gs_effect_get_param_by_name(f->effect, "depth_tex");
		f->p_showdepth = gs_effect_get_param_by_name(f->effect, "show_depth");
		f->p_depthtexel = gs_effect_get_param_by_name(f->effect, "depth_texel");
		f->p_eyefit = gs_effect_get_param_by_name(f->effect, "eye_fit");
		f->p_dither = gs_effect_get_param_by_name(f->effect, "dither");
		f->p_outsize = gs_effect_get_param_by_name(f->effect, "out_size");
		f->p_debug = gs_effect_get_param_by_name(f->effect, "debug_tint");
	}
	f->rt_full = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	f->rt_pre[0] = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	f->rt_pre[1] = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	f->rt_small = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	f->stage[0] = gs_stagesurface_create(f->infer_w, f->infer_h, GS_RGBA);
	f->stage[1] = gs_stagesurface_create(f->infer_w, f->infer_h, GS_RGBA);
	std::vector<float> half((size_t)f->infer_w * f->infer_h, 0.5f);
	f->depth_flat = gs_texture_create(f->infer_w, f->infer_h, GS_R32F, 1,
					  nullptr, GS_DYNAMIC);
	gs_texture_set_image(f->depth_flat, (const uint8_t *)half.data(),
			     f->infer_w * sizeof(float), false);
	obs_leave_graphics();

	bfree(effect_path);
	bfree(model_path);

	if (f->ort_ok)
		f->worker = std::thread(worker_fn, f);

	/* How much of the parent's persisted sync offset is our leftover audio delay
	 * from last session (written by real3d_save). The first render subtracts it so
	 * we recover the user's true baseline instead of compounding on it. Mirror it
	 * into sync_leak_ns so that if OBS saves *before* that first render (source
	 * hidden / not yet drawn), real3d_save re-persists the leak instead of 0 --
	 * otherwise the subtraction info would be lost and the offset stuck inflated. */
	f->load_leak_ns = obs_data_get_int(settings, "sync_leak_ns");
	f->sync_leak_ns.store(f->load_leak_ns, std::memory_order_relaxed);

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
	free_depth_cache(f);
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
	if (f->depth_flat)
		gs_texture_destroy(f->depth_flat);
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
	/* Built dynamically (vs the static REAL3D_BUILD_INFO macro) so the active
	 * inference size is shown right after the version: the dims come from the
	 * loaded model (per-filter f->infer_w/h), letting the user tell the full vs
	 * lite (392x224) build apart in the UI, not just the log. */
	std::string bi = std::string("near Real 3D ") + REAL3D_VERSION;
	if (f && f->ort_ok)
		bi += "  [model " + std::to_string(f->infer_w) + "x" +
		      std::to_string(f->infer_h) + "]";
	else if (f)
		bi += "  [model: none -> luminance]";
	bi += "  (built " __DATE__ " " __TIME__ ")";
	obs_properties_add_text(p, "build_info", bi.c_str(), OBS_TEXT_INFO);

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
	obs_property_set_long_description(tier, obs_module_text("strength.desc"));
	q = obs_properties_add_float_slider(g3d, "convergence",
					    obs_module_text("convergence"), 0.0,
					    1.0, 0.01);
	obs_property_set_long_description(q, obs_module_text("convergence.desc"));
	q = obs_properties_add_float_slider(g3d, "grading",
					    obs_module_text("grading"), 0.0, 1.0,
					    0.05);
	obs_property_set_long_description(q, obs_module_text("grading.desc"));
	q = obs_properties_add_float_slider(g3d, "silhouette",
					    obs_module_text("silhouette"), 0.0,
					    1.0, 0.05);
	obs_property_set_long_description(q, obs_module_text("silhouette.desc"));
	q = obs_properties_add_bool(g3d, "swap", obs_module_text("swap"));
	obs_property_set_long_description(q, obs_module_text("swap.desc"));
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
	q = obs_properties_add_float_slider(g3d, "dither",
					    obs_module_text("dither"), 0.0, 100.0,
					    1.0);
	obs_property_set_long_description(q, obs_module_text("dither.desc"));
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
	q = obs_properties_add_bool(g_temp, "auto_suppress",
				    obs_module_text("autosuppress"));
	obs_property_set_long_description(q, obs_module_text("autosuppress.desc"));
	q = obs_properties_add_group(gst, "temporal", obs_module_text("temporal"),
				     OBS_GROUP_CHECKABLE, g_temp);
	obs_property_set_long_description(q, obs_module_text("temporal.desc"));

	q = obs_properties_add_float_slider(gst, "depth_smooth",
					    obs_module_text("edgesoft"), 0.0, 1.0,
					    0.05);
	obs_property_set_long_description(q, obs_module_text("edgesoft.desc"));
	/* Frame-matched delay is the parent toggle; the margin only matters while it
	 * is on, so it sits inside the checkable group and is disabled in place when
	 * delay is off (same idiom as skip_static / temporal above). The group's
	 * checkbox is stored under "sync_delay" -- same key as the old bool, so
	 * update()/defaults() are unchanged. */
	obs_properties_t *g_delay = obs_properties_create();
	q = obs_properties_add_int_slider(g_delay, "delay_margin",
					  obs_module_text("delaymargin"), 0, 4, 1);
	obs_property_set_long_description(q, obs_module_text("delaymargin.desc"));
	q = obs_properties_add_group(gst, "sync_delay", obs_module_text("syncdelay"),
				     OBS_GROUP_CHECKABLE, g_delay);
	obs_property_set_long_description(q, obs_module_text("syncdelay.desc"));
	obs_properties_add_group(p, "grp_stab", obs_module_text("group.stab"),
				 OBS_GROUP_NORMAL, gst);

	/* ---- debug ---- */
	obs_properties_t *gdbg = obs_properties_create();
	q = obs_properties_add_bool(gdbg, "show_depth",
				    obs_module_text("showdepth"));
	obs_property_set_long_description(q, obs_module_text("showdepth.desc"));
	q = obs_properties_add_bool(gdbg, "debug_overlay",
				    obs_module_text("debugoverlay"));
	obs_property_set_long_description(q, obs_module_text("debugoverlay.desc"));
	q = obs_properties_add_bool(gdbg, "debug_cache1",
				    obs_module_text("debugcache1"));
	obs_property_set_long_description(q, obs_module_text("debugcache1.desc"));
	obs_properties_add_group(p, "grp_dbg", obs_module_text("group.debug"),
				 OBS_GROUP_NORMAL, gdbg);

	return p;
}

static void real3d_defaults(obs_data_t *s)
{
	obs_data_set_default_int(s, "strength", 2);
	obs_data_set_default_double(s, "convergence", 0.5);
	obs_data_set_default_double(s, "grading", 0.0);
	obs_data_set_default_double(s, "silhouette", 0.35);
	obs_data_set_default_bool(s, "swap", false);
	obs_data_set_default_bool(s, "full_sbs", true);
	obs_data_set_default_int(s, "sbs_size", 0);
	obs_data_set_default_bool(s, "eye_letterbox", false);
	obs_data_set_default_double(s, "dither", 12.0);
	obs_data_set_default_bool(s, "auto_suppress", true);
	obs_data_set_default_int(s, "infer_fps", 15);
	obs_data_set_default_bool(s, "skip_static", true);
	obs_data_set_default_double(s, "static_thresh", 1.0);
	obs_data_set_default_bool(s, "temporal", true);
	obs_data_set_default_double(s, "stabilize_strength", 0.4);
	obs_data_set_default_double(s, "depth_smooth", 0.3);
	obs_data_set_default_bool(s, "sync_delay", false);
	obs_data_set_default_int(s, "delay_margin", 2);
	obs_data_set_default_bool(s, "show_depth", false);
	obs_data_set_default_bool(s, "debug_overlay", false);
	obs_data_set_default_bool(s, "debug_cache1", false);
}

/* Persist how much of the parent's (also-persisted) audio sync offset is our
 * delay, so the next load can subtract it and recover the user's true baseline
 * instead of compounding on it. Called by OBS when saving the scene collection. */
static void real3d_save(void *data, obs_data_t *settings)
{
	auto *f = static_cast<real3d_filter *>(data);
	obs_data_set_int(settings, "sync_leak_ns",
			 f->sync_leak_ns.load(std::memory_order_relaxed));
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

	/* Area-average downscale to the inference size (infer_w x infer_h) via a
	 * progressive 2:1 halving pyramid (a hand-rolled mipmap; libobs exposes no
	 * runtime mip-gen and texrender can't mip). A single steep bilinear reduction
	 * only reads a 2x2 footprint, so reducing by more than 2:1 (e.g. a 1920/4K
	 * source in one step) under-samples and lets aliasing + compression banding
	 * through. Halving repeatedly box-averages the whole footprint instead; the
	 * last step reduces to infer_w x infer_h, by which point each axis reduces by
	 * <=~2:1 so the 2x2 bilinear tap is sufficient. Inference path only -- the
	 * visible warp still samples the full-res frame, so sharpness is unaffected. */
	gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	gs_eparam_t *dimg = gs_effect_get_param_by_name(def, "image");
	struct vec4 clr;
	vec4_zero(&clr);
	const uint32_t PRE_W = f->infer_w * 2, PRE_H = f->infer_h * 2;

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
	if (!gs_texrender_begin(f->rt_small, f->infer_w, f->infer_h))
		return;
	gs_clear(GS_CLEAR_COLOR, &clr, 0.0f, 0);
	gs_ortho(0.0f, (float)f->infer_w, 0.0f, (float)f->infer_h, -100.0f, 100.0f);
	gs_effect_set_texture(dimg, src);
	while (gs_effect_loop(def, "Draw"))
		gs_draw_sprite(src, 0, f->infer_w, f->infer_h);
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
	f->readback.resize((size_t)f->infer_w * f->infer_h * 4);
	std::vector<uint8_t> &tight = f->readback;
	for (int y = 0; y < f->infer_h; ++y)
		memcpy(&tight[(size_t)y * f->infer_w * 4],
		       data + (size_t)y * linesize,
		       (size_t)f->infer_w * 4);
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
		 * hold is derived at warp time from "no cached depth matches the shown
		 * frame", so it engages exactly when an unmatched post-cut frame reaches
		 * the screen (which, in delay mode, is several frames after this). */
		f->scene_gen++;
		/* Cut detection lags by ~1 frame (the mapped readback is one detect-tick
		 * old), so new-scene frames captured before this point are already in the
		 * delay ring tagged with the *old* generation. Re-tag every ring frame at
		 * or after the detected cut's capture time so the delayed display still
		 * pairs them with the new scene's depth (unfilled slots have ts 0 < any
		 * real ts, so they're left alone). Empty ring (delay off) -> no-op. */
		for (int i = 0; i < (int)f->ring.size(); ++i)
			if (f->ring_ts[i] >= readback_ts)
				f->ring_gen[i] = f->scene_gen;
		if (f->debug_overlay.load(std::memory_order_relaxed))
			blog(LOG_INFO, "[near-real3d] cut detected -> gen %u "
				       "(mad %.0f) [delay %d = commit %d + margin %d, "
				       "cache %d]",
			     f->scene_gen, cut_mad, f->applied_delay,
			     f->commit_delay,
			     f->delay_margin.load(std::memory_order_relaxed),
			     f->depth_used);
		f->settle_left = SETTLE_FRAMES;
		f->prev_in = tight; /* fresh static-skip baseline for the new scene */
		f->last_submit_ns = now;
		{
			std::lock_guard<std::mutex> lk(f->m);
			f->in_buf = tight; /* overwrite any unconsumed queued (stale) frame */
			f->in_gen = f->scene_gen;
			f->in_ts = readback_ts;
			f->in_submit_ns = now;
			f->in_forced = true; /* cut-forced: keep its latency out of the EMA */
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

	/* Static-frame skip: reuse the depth already cached when this frame
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
		f->in_submit_ns = now;
		f->in_forced = false; /* steady cadence sample: feeds the latency EMA */
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

	/* Canvas frame interval (ns): drives the delay-frame commit, the audio offset,
	 * and the depth-cache sizing. Computed once here so the depth upload below can
	 * size the cache before the delay block runs. */
	uint64_t interval_ns = 0;
	{
		struct obs_video_info ovi;
		if (obs_get_video_info(&ovi) && ovi.fps_num)
			interval_ns = (uint64_t)ovi.fps_den * 1000000000ULL /
				      (uint64_t)ovi.fps_num;
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
		uint64_t got_wait = 0, got_compute = 0;
		bool got_forced = false;
		{
			std::lock_guard<std::mutex> lk(f->m);
			if (f->depth_ready) {
				got.swap(f->depth_buf);
				got_gen = f->out_gen;
				got_ts = f->out_ts;
				got_wait = f->out_wait_ns;
				got_compute = f->out_compute_ns;
				f->scene_conf = f->out_conf; /* latest auto-suppress target */
				got_forced = f->out_forced; /* cut-forced -> skip the latency EMA */
				f->depth_ready = false;
			}
		}
		/* Size the depth cache every frame so the delay-driven slot count and the
		 * debug 1-slot toggle take effect promptly. depth_used is the effective
		 * live window (<= allocated): the debug toggle forces it to 1, reproducing
		 * the pre-cache "latest depth only" behaviour for an instant A/B without
		 * reallocating (allocation is append-only; we just read/write fewer slots). */
		const int want_slots =
			f->debug_cache1.load(std::memory_order_relaxed)
				? 1
				: wanted_depth_slots(f, interval_ns);
		ensure_depth_cache(f, want_slots);
		const int cap = (int)f->depth_cache.size();
		f->depth_used = want_slots < cap ? want_slots : cap;
		if (f->depth_used < 1)
			f->depth_used = 1;
		if (f->depth_widx >= f->depth_used)
			f->depth_widx = 0;

		if (!got.empty() && !f->depth_cache.empty()) {
			/* Store into the next live slot (round-robin), tagged with the
			 * scene generation and source capture timestamp it was computed for,
			 * so the warp can later pick the depth matching the frame it shows. */
			depth_slot &s = f->depth_cache[f->depth_widx];
			gs_texture_set_image(s.tex, (const uint8_t *)got.data(),
					     f->infer_w * sizeof(float), false);
			s.gen = got_gen;
			s.ts = got_ts;
			s.valid = true;
			f->depth_widx = (f->depth_widx + 1) % f->depth_used;
		}
		if (!got.empty()) {
			/* Measure the capture->depth pipeline latency (LPF). This is how
			 * far the image must be delayed in sync mode to match the depth.
			 * Only measure while sync delay is ON: the EMA is unused otherwise,
			 * and measuring with it off lets a hide/resume from any duration
			 * (which the tick reset and render-gap can't always cover when we
			 * don't own sync) poison the value for a later enable. Cut-forced
			 * (off-cadence) depths are also excluded (!got_forced): their
			 * capture->depth timing isn't representative of the steady cadence,
			 * so feeding them in made commit_delay -- and thus the audio sync
			 * offset -- hunt by a frame on every scene change (audible re-time). */
			if (f->sync_delay.load(std::memory_order_relaxed) && !got_forced &&
			    got_ts && got_ts >= f->latency_epoch_ns && now > got_ts) {
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
				     f->infer_w, f->infer_h);
			}
			/* Latency decomposition for the *first* depth of each generation
			 * (post-cut that is the forced inference): shows where the cut-path
			 * latency over `commit` goes. total = now - source capture ts =
			 * stage/detect lag + queue wait + compute + render pickup; `other`
			 * is the stage lag + pickup (everything outside the worker). */
			if (got_gen != f->last_logged_depth_gen &&
			    f->debug_overlay.load(std::memory_order_relaxed) &&
			    got_ts && now > got_ts) {
				f->last_logged_depth_gen = got_gen;
				const double iv = interval_ns
							  ? (double)interval_ns / 1e6
							  : 16.667;
				const double total = (double)(now - got_ts) / 1e6;
				const double wait = (double)got_wait / 1e6;
				const double comp = (double)got_compute / 1e6;
				blog(LOG_INFO,
				     "[near-real3d] depth gen %u latency %.1f ms "
				     "(%.1f frames) = wait %.1f + compute %.1f + "
				     "other %.1f | commit %d (%.1f ms)",
				     got_gen, total, total / iv, wait, comp,
				     total - wait - comp, f->commit_delay,
				     (double)f->commit_delay * iv);
			}
		}
	}

	/* Frame-matched delay: pick which buffered frame to warp, and delay the
	 * source audio to match. When the mode is off, release the audio sync and
	 * free the ring. shown_gen / shown_ts identify whatever frame we end up
	 * warping (the delayed ring frame, or the current one when not delaying);
	 * the warp uses them below to pick the matching cached depth. */
	gs_texture_t *show = full;
	uint32_t shown_gen = f->scene_gen;
	uint64_t shown_ts = now;
	{
		/* One-time on-load correction (runs regardless of delay state): OBS
		 * persisted our last-session audio delay into the parent's sync offset.
		 * Subtract what real3d_save recorded so the user's true baseline is
		 * restored and we never compound on it. We only ever ADD a non-negative
		 * delay, so we subtract straight (no clamp): this also recovers a
		 * legitimately NEGATIVE user baseline (OBS allows down to -950 ms). Run
		 * only once we can reach the parent so the leak isn't dropped if the very
		 * first render happens before the source is attached. */
		if (f->load_leak_pending) {
			obs_source_t *parent = obs_filter_get_parent(f->context);
			if (parent) {
				f->load_leak_pending = false;
				if (f->load_leak_ns > 0) {
					const int64_t cur =
						obs_source_get_sync_offset(parent);
					obs_source_set_sync_offset(parent,
								   cur - f->load_leak_ns);
					if (f->debug_overlay.load(std::memory_order_relaxed))
						blog(LOG_INFO,
						     "[near-real3d] audio leak corrected on "
						     "load: %.1f -> %.1f ms (removed %.1f)",
						     (double)cur / 1e6,
						     (double)(cur - f->load_leak_ns) / 1e6,
						     (double)f->load_leak_ns / 1e6);
				}
				/* Our leak is no longer in the parent offset. Store 0 AFTER the
				 * offset write so a torn save errs toward over-removal (bounded,
				 * non-compounding) rather than leaving the leak in. */
				f->sync_leak_ns.store(0, std::memory_order_relaxed);
				f->load_leak_ns = 0;
			}
		}

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
					/* DIAGNOSTIC: the offset we read here is treated as the
					 * user's baseline. If it is non-zero on a fresh start (or
					 * grows across restarts after resetting it to 0), our delay
					 * leaked into OBS's persisted "sync" and is compounding. */
					if (f->debug_overlay.load(std::memory_order_relaxed))
						blog(LOG_INFO,
						     "[near-real3d] audio sync acquired: "
						     "parent offset at takeover = %.1f ms "
						     "(used as baseline)",
						     (double)f->saved_sync / 1e6);
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
			/* Delay by the measured latency plus the margin (shared with the
			 * depth-cache sizing) so the post-cut depth reliably lands before the
			 * new-scene frames reach the screen -- otherwise their leading edge
			 * briefly has no matching depth (a short 2D flash). Audio follows via
			 * applied_d below, so it stays in sync. */
			const int want = wanted_delay_frames(f);

			ensure_delay_ring(f, w, h, want + 3);
			const int n = (int)f->ring.size();
			int applied_d = 0; /* frames the image is actually delayed by */
			if (n >= 2) {
				int d = want;
				if (d > n - 1)
					d = n - 1;
				gs_copy_texture(f->ring[f->ring_widx], full);
				f->ring_gen[f->ring_widx] = f->scene_gen;
				f->ring_ts[f->ring_widx] = now;
				if (d > 0 && f->ring_filled >= d) {
					const int idx =
						(f->ring_widx - d + n) % n;
					show = f->ring[idx];
					shown_gen = f->ring_gen[idx];
					shown_ts = f->ring_ts[idx];
					applied_d = d;
				}
				f->ring_widx = (f->ring_widx + 1) % n;
				if (f->ring_filled < n)
					f->ring_filled++;
			}
			f->applied_delay = applied_d; /* for the debug log */

			/* Audio delay = the frames the image is actually delayed (0 during
			 * ring warm-up). Because `applied_d` is driven by the committed,
			 * deadbanded delay it only changes on a genuine, sustained latency
			 * shift -- so we re-apply (and OBS re-times) at most rarely. */
			const int64_t extra =
				interval_ns
					? (int64_t)applied_d * (int64_t)interval_ns
					: 0;
			if (f->sync_owned && extra != f->applied_extra) {
				/* Record our leak BEFORE writing the parent offset. If a
				 * scene-collection save (UI thread) interleaves between the two
				 * writes, the persisted leak is then never *smaller* than the
				 * leak baked into the persisted offset, so the next load
				 * over-removes (bounded harmless by the load-time clamp) instead
				 * of under-removing -- the latter would re-introduce compounding.
				 * Release uses the mirror order (restore offset, then zero leak)
				 * for the same safe-direction reason. */
				f->sync_leak_ns.store(extra, std::memory_order_relaxed);
				obs_source_t *parent =
					obs_filter_get_parent(f->context);
				if (parent)
					obs_source_set_sync_offset(
						parent, f->saved_sync + extra);
				if (f->debug_overlay.load(std::memory_order_relaxed))
					blog(LOG_INFO,
					     "[near-real3d] audio offset set: baseline %.1f "
					     "+ extra %.1f = %.1f ms (applied_d %d frames)",
					     (double)f->saved_sync / 1e6,
					     (double)extra / 1e6,
					     (double)(f->saved_sync + extra) / 1e6, applied_d);
				f->applied_extra = extra;
			}
		} else {
			release_audio_sync(f);
			if (!f->ring.empty())
				free_delay_ring(f);
			f->applied_delay = 0; /* not delaying */
		}
	}

	/* Pick the cached depth matching the frame we are about to warp (same scene
	 * generation, nearest capture timestamp). In delay mode this keeps the prior
	 * scene's tail in 3D across a cut -- its depth is still cached for the prior-
	 * scene frames the ring is still showing -- and only the genuinely unmatched
	 * frames (no depth for that scene yet: pre-warm-up, a forced post-cut depth
	 * not landed, continuous cuts) fall back to 2D. */
	bool depth_found = false;
	gs_texture_t *depth_show = pick_depth(f, shown_gen, shown_ts, depth_found);

	/* Scene-cut disparity fade. Flatten to 2D whenever the frame we are about to
	 * warp has no matching cached depth -- warping it with another scene's depth
	 * is the glitch we are avoiding. Once a match exists, ramp back to full over
	 * ~1/DISP_RAMP_PER_SEC s so the 3D returns without a pop. A failsafe lifts the
	 * hold if a match never appears (e.g. inference failure) so the filter can't
	 * get stuck flat. */
	{
		const bool dbg = f->debug_overlay.load(std::memory_order_relaxed);
		/* Only the ONNX depth path uses the cache; in the luminance fallback
		 * (model missing / init or runtime failure) the shader derives depth from
		 * the image itself (use_depth_tex=0), so a cache miss must NOT flatten to
		 * 2D there -- otherwise startup with a broken model shows flat until the
		 * failsafe lifts. Treat "stale" as meaningful only while ONNX is live. */
		const bool depth_active =
			f->ort_ok && f->ort_live.load(std::memory_order_relaxed);
		const bool depth_stale = depth_active && !depth_found;
		if (depth_stale) {
			if (!f->stale_since_ns) {
				f->stale_since_ns = now; /* flat hold begins */
				f->stale_logged = false;
			}
		} else {
			/* Flat hold ended by a genuine match: log how long the 2D
			 * fallback (red) lasted and how close the recovering depth was. */
			if (f->stale_since_ns && dbg)
				blog(LOG_INFO, "[near-real3d] flat hold ended: %.1f ms "
					       "(gen %u, depth ts_delta %.1f ms) "
					       "[delay %d = commit %d + margin %d, "
					       "cache %d]",
				     (double)(now - f->stale_since_ns) / 1e6,
				     shown_gen, (double)f->depth_match_dt / 1e6,
				     f->applied_delay, f->commit_delay,
				     f->delay_margin.load(std::memory_order_relaxed),
				     f->depth_used);
			f->stale_since_ns = 0;
		}
		const bool failsafe =
			f->stale_since_ns &&
			now - f->stale_since_ns > FLATTEN_TIMEOUT_NS;
		if (failsafe && dbg && !f->stale_logged) {
			blog(LOG_WARNING, "[near-real3d] flat hold > %.0f ms without "
					  "matching depth -> failsafe lifting (gen %u) "
					  "[delay %d = commit %d + margin %d, cache %d]",
			     (double)FLATTEN_TIMEOUT_NS / 1e6, shown_gen,
			     f->applied_delay, f->commit_delay,
			     f->delay_margin.load(std::memory_order_relaxed),
			     f->depth_used);
			f->stale_logged = true;
		}
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
		/* Ease the auto-suppression multiplier toward the worker's latest scene
		 * confidence (or 1 when the feature is off) so the disparity glides
		 * rather than steps when a new depth lands. The confidence is already
		 * time-smoothed in the stabiliser; this is just the render-rate glue. */
		{
			const float target =
				(f->auto_suppress.load(std::memory_order_relaxed) &&
				 f->ort.temporal.load(std::memory_order_relaxed))
					? f->scene_conf
					: 1.0f;
			const float step = CONF_RAMP_PER_SEC * dt;
			if (f->conf_scale < target) {
				f->conf_scale += step;
				if (f->conf_scale > target)
					f->conf_scale = target;
			} else if (f->conf_scale > target) {
				f->conf_scale -= step;
				if (f->conf_scale < target)
					f->conf_scale = target;
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
		gs_effect_set_texture_srgb(f->p_image, show);
	else
		gs_effect_set_texture(f->p_image, show);
	const auto rel = std::memory_order_relaxed;
	const bool full_sbs = f->full_sbs.load(rel);
	gs_effect_set_texture(f->p_depthtex, depth_show);
	gs_effect_set_float(f->p_strength,
			    f->frac.load(rel) * f->disp_scale * f->conf_scale);
	gs_effect_set_float(f->p_conv, f->convergence.load(rel));
	gs_effect_set_float(f->p_grading, f->grading.load(rel));
	gs_effect_set_float(f->p_silhouette, f->silhouette.load(rel));
	gs_effect_set_float(f->p_swap, f->swap_sign.load(rel));
	gs_effect_set_float(f->p_usedepth,
			    (f->ort_ok && f->ort_live.load(rel)) ? 1.0f : 0.0f);
	gs_effect_set_float(f->p_showdepth, f->show_depth.load(rel) ? 1.0f : 0.0f);
	struct vec2 dtexel;
	dtexel.x = f->infer_w > 0 ? 1.0f / (float)f->infer_w : 1.0f;
	dtexel.y = f->infer_h > 0 ? 1.0f / (float)f->infer_h : 1.0f;
	gs_effect_set_vec2(f->p_depthtexel, &dtexel);

	/* Debug overlay: only meaningful while the ONNX depth path is live (in the
	 * luminance fallback the warp is 3D regardless of the cache, so the match
	 * state would mislead). 1 = no matching cached depth (red, 2D fallback),
	 * 2 = matched but disparity still ramping (yellow), 0 = full 3D / off. */
	float dbg = 0.0f;
	if (f->debug_overlay.load(rel) && f->ort_ok && f->ort_live.load(rel)) {
		if (!depth_found)
			dbg = 1.0f;
		else if (f->disp_scale < 0.999f)
			dbg = 2.0f;
	}
	if (f->p_debug)
		gs_effect_set_float(f->p_debug, dbg);

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
	/* Output dither: seeded from the output pixel, so the shader needs the SBS
	 * output dims. Applied at the warp's final 8-bit write to mask the banding
	 * its own re-quantization would re-create (see near-real3d.effect). */
	gs_effect_set_float(f->p_dither, f->dither.load(rel));
	struct vec2 osz;
	osz.x = (float)out_w;
	osz.y = (float)out_h;
	gs_effect_set_vec2(f->p_outsize, &osz);
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
 * An mpv-style deband as a plain 1:1 image filter, so it can sit on any source /
 * anywhere in a filter chain -- not only on the 3D filter. Runs at the source
 * resolution (taps + grain in true source pixels; grain added at output
 * resolution = best 8-bit dither). SRGB is handled by libobs (OBS_SOURCE_SRGB)
 * like obs-filters' sharpness_v2, so it samples in linear light. For 3D use,
 * place this above the SBS filter: it debands/dithers the source the warp then
 * samples (the warp's 8-bit re-sampling can still leave faint residual banding,
 * which the grain masks). The 3D filter warns if a deband sits below it.
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
	real3d_filter_info.save = real3d_save;
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
