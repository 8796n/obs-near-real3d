// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 8796n <info@8796.jp>
/*
 * Flow-guided temporal stabiliser + normaliser for obs-near-real3d.
 *
 * Owns all temporal state (so scene cuts can reset it in one place):
 *   1. range normalisation of the raw depth (EMA of min/max) so the depth->[0,1]
 *      mapping doesn't breathe frame to frame;
 *   2. flow-guided per-pixel temporal smoothing -- single-image depth models
 *      (DA V2) have no temporal consistency, so when anything moves the depth of
 *      static regions also wobbles. We dense-flow the previous stabilised depth
 *      into the current frame and blend, so static AND tracked-moving regions are
 *      smoothed; large residuals (dis-occlusion / flow fail) fall back to the
 *      fresh depth;
 *   3. optional history clipping + reactive rejection -- clamp warped history to
 *      the current local depth range, and fully discard history near fast motion
 *      or uncertain flow so thin movers do not leave depth trails;
 *   4. scene-cut detection -- on a hard cut the flow is meaningless and the EMA
 *      scale lags ~0.7 s, so we detect the cut (mean frame diff) and snap both
 *      the range and the blend to the fresh frame (recovers in one depth frame);
 *   5. depth-edge softening (Gaussian) to reduce DIBR rubber-sheet tearing.
 *
 * Self-contained: the dense optical flow is a small pyramidal Lucas-Kanade
 * (no OpenCV). The depth map is low-frequency and the residual fallback masks
 * flow errors, so an approximate flow is sufficient. Worker-thread only.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace nr3d {

/* bilinear sample with clamped (replicate) borders */
inline float bilinear(const std::vector<float> &im, int w, int h, float x,
		      float y)
{
	if (x < 0.f)
		x = 0.f;
	if (y < 0.f)
		y = 0.f;
	if (x > w - 1.f)
		x = w - 1.f;
	if (y > h - 1.f)
		y = h - 1.f;
	const int x0 = (int)x, y0 = (int)y;
	const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
	const float fx = x - x0, fy = y - y0;
	const float a = im[y0 * w + x0], b = im[y0 * w + x1];
	const float c = im[y1 * w + x0], d = im[y1 * w + x1];
	return (a * (1 - fx) + b * fx) * (1 - fy) +
	       (c * (1 - fx) + d * fx) * fy;
}

/* 2x2 box-average downsample (odd sizes clamp the last sample) */
inline void downsample2(const std::vector<float> &s, int w, int h,
			std::vector<float> &d, int &dw, int &dh)
{
	dw = (w + 1) / 2;
	dh = (h + 1) / 2;
	d.assign((size_t)dw * dh, 0.f);
	for (int y = 0; y < dh; ++y) {
		const int y0 = std::min(2 * y, h - 1), y1 = std::min(2 * y + 1, h - 1);
		for (int x = 0; x < dw; ++x) {
			const int x0 = std::min(2 * x, w - 1),
				  x1 = std::min(2 * x + 1, w - 1);
			d[y * dw + x] = 0.25f * (s[y0 * w + x0] + s[y0 * w + x1] +
						 s[y1 * w + x0] + s[y1 * w + x1]);
		}
	}
}

/* windowed SUM over a (2r+1)^2 box, replicate borders, separable running sum */
inline void boxSum(std::vector<float> &im, int w, int h, int r)
{
	std::vector<float> tmp((size_t)w * h);
	auto cl = [](int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); };
	/* horizontal */
	for (int y = 0; y < h; ++y) {
		const float *row = &im[(size_t)y * w];
		float *out = &tmp[(size_t)y * w];
		float acc = 0.f;
		for (int k = -r; k <= r; ++k)
			acc += row[cl(k, w)];
		out[0] = acc;
		for (int x = 1; x < w; ++x) {
			acc += row[cl(x + r, w)] - row[cl(x - 1 - r, w)];
			out[x] = acc;
		}
	}
	/* vertical */
	for (int x = 0; x < w; ++x) {
		float acc = 0.f;
		for (int k = -r; k <= r; ++k)
			acc += tmp[(size_t)cl(k, h) * w + x];
		im[(size_t)0 * w + x] = acc;
		for (int y = 1; y < h; ++y) {
			acc += tmp[(size_t)cl(y + r, h) * w + x] -
			       tmp[(size_t)cl(y - 1 - r, h) * w + x];
			im[(size_t)y * w + x] = acc;
		}
	}
}

/* separable normalised Gaussian blur, replicate borders */
inline void gaussianBlur(std::vector<float> &im, int w, int h, float sigma)
{
	if (sigma <= 0.f)
		return;
	int r = (int)std::ceil(3.f * sigma);
	if (r < 1)
		r = 1;
	std::vector<float> k(2 * r + 1);
	float sum = 0.f;
	const float inv2s2 = 1.f / (2.f * sigma * sigma);
	for (int i = -r; i <= r; ++i) {
		float v = std::exp(-(float)(i * i) * inv2s2);
		k[i + r] = v;
		sum += v;
	}
	for (float &v : k)
		v /= sum;
	auto cl = [](int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); };
	std::vector<float> tmp((size_t)w * h);
	for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x) {
			float acc = 0.f;
			for (int i = -r; i <= r; ++i)
				acc += k[i + r] * im[(size_t)y * w + cl(x + i, w)];
			tmp[(size_t)y * w + x] = acc;
		}
	for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x) {
			float acc = 0.f;
			for (int i = -r; i <= r; ++i)
				acc += k[i + r] * tmp[(size_t)cl(y + i, h) * w + x];
			im[(size_t)y * w + x] = acc;
		}
}

/* separable max filter over a (2r+1)^2 window, replicate borders */
inline void maxFilter(std::vector<float> &im, int w, int h, int r)
{
	if (r < 1)
		return;
	auto cl = [](int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); };
	std::vector<float> tmp((size_t)w * h);
	for (int y = 0; y < h; ++y) {
		const float *row = &im[(size_t)y * w];
		float *out = &tmp[(size_t)y * w];
		for (int x = 0; x < w; ++x) {
			float m = row[x];
			for (int k = -r; k <= r; ++k)
				m = std::max(m, row[cl(x + k, w)]);
			out[x] = m;
		}
	}
	for (int x = 0; x < w; ++x)
		for (int y = 0; y < h; ++y) {
			float m = tmp[(size_t)y * w + x];
			for (int k = -r; k <= r; ++k)
				m = std::max(m, tmp[(size_t)cl(y + k, h) * w + x]);
			im[(size_t)y * w + x] = m;
		}
}

/* Edge-localised depth softening. A plain global blur rounds off *all* depth
 * detail and bleeds the foreground silhouette outward; here we feather only the
 * depth discontinuities, where the backward warp tears (DIBR rubber-sheet),
 * and leave flat/smoothly-varying regions crisp. We build a per-pixel weight
 * from the depth-gradient magnitude (flat = 0, steep edge = 1), dilate it to
 * cover the blur's footprint so the whole transition band feathers at full
 * strength (not just the one-pixel gradient spike), soften the weight boundary,
 * then blend the depth toward a blurred copy by that weight. */
inline void edgeSoftenDepth(std::vector<float> &depth, int w, int h, float sigma,
			    float edge_lo, float edge_hi)
{
	if (sigma <= 0.f)
		return;
	const int n = w * h;
	if ((int)depth.size() != n || edge_hi <= edge_lo)
		return;

	std::vector<float> wt((size_t)n);
	const float inv_span = 1.f / (edge_hi - edge_lo);
	for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x) {
			const int xm = x > 0 ? x - 1 : 0;
			const int xp = x < w - 1 ? x + 1 : w - 1;
			const int ym = y > 0 ? y - 1 : 0;
			const int yp = y < h - 1 ? y + 1 : h - 1;
			const float gx = 0.5f * (depth[(size_t)y * w + xp] -
						 depth[(size_t)y * w + xm]);
			const float gy = 0.5f * (depth[(size_t)yp * w + x] -
						 depth[(size_t)ym * w + x]);
			float t = (std::sqrt(gx * gx + gy * gy) - edge_lo) * inv_span;
			t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
			wt[(size_t)y * w + x] = t * t * (3.f - 2.f * t); /* smoothstep */
		}

	/* Dilate the weight across the blurred copy's *full* footprint
	 * (ceil(3*sigma) = gaussianBlur's own radius). A narrower band left a
	 * residual depth step just outside it -- the warp then concentrated the
	 * disocclusion stretch into that narrow band and the silhouette looked
	 * sharply distorted. Covering the whole ramp makes the edge feather as
	 * softly as a global blur would, while flat regions stay crisp. */
	int r = (int)std::ceil(3.f * sigma);
	if (r < 1)
		r = 1;
	maxFilter(wt, w, h, r);            /* cover the feather band */
	gaussianBlur(wt, w, h, sigma * 0.5f); /* soften the weight boundary */

	std::vector<float> blurred = depth;
	gaussianBlur(blurred, w, h, sigma);
	for (int i = 0; i < n; ++i) {
		float ww = wt[i] < 0.f ? 0.f : (wt[i] > 1.f ? 1.f : wt[i]);
		depth[i] += ww * (blurred[i] - depth[i]);
	}
}

/* Clamp history to the range represented by the current 3x3 neighbourhood.
 * This removes stale foreground depth where an object has already moved away. */
inline void clipHistory3x3(const std::vector<float> &cur,
			   const std::vector<float> &history, int w, int h,
			   std::vector<float> &clipped)
{
	clipped.resize(history.size());
	for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x) {
			float lo = cur[(size_t)y * w + x];
			float hi = lo;
			for (int dy = -1; dy <= 1; ++dy)
				for (int dx = -1; dx <= 1; ++dx) {
					const int xx = std::clamp(x + dx, 0, w - 1);
					const int yy = std::clamp(y + dy, 0, h - 1);
					const float v = cur[(size_t)yy * w + xx];
					lo = std::min(lo, v);
					hi = std::max(hi, v);
				}
			const size_t i = (size_t)y * w + x;
			clipped[i] = std::clamp(history[i], lo, hi);
		}
}

/* Expand unreliable-history pixels to cover antialiased / softened edges. */
inline void dilateMask(std::vector<uint8_t> &mask, int w, int h, int radius)
{
	const std::vector<uint8_t> src = mask;
	for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x) {
			if (!src[(size_t)y * w + x])
				continue;
			for (int dy = -radius; dy <= radius; ++dy)
				for (int dx = -radius; dx <= radius; ++dx) {
					const int xx = std::clamp(x + dx, 0, w - 1);
					const int yy = std::clamp(y + dy, 0, h - 1);
					mask[(size_t)yy * w + xx] = 1;
				}
		}
}

/* Mild per-channel blur of an RGBA8 buffer in place (inference-input only):
 * smooths compression banding / block noise so the depth model doesn't read
 * tone jumps as depth steps. Reuses the separable Gaussian above. The visible
 * warp samples the full-res frame, so this never softens the output image. */
inline void smoothRGBA(uint8_t *rgba, int w, int h, float sigma)
{
	if (sigma <= 0.f)
		return;
	const int n = w * h;
	std::vector<float> ch((size_t)n);
	for (int c = 0; c < 3; ++c) { /* R,G,B; leave alpha */
		for (int i = 0; i < n; ++i)
			ch[i] = (float)rgba[(size_t)i * 4 + c];
		gaussianBlur(ch, w, h, sigma);
		for (int i = 0; i < n; ++i) {
			float v = ch[i] + 0.5f;
			rgba[(size_t)i * 4 + c] =
				(uint8_t)(v < 0.f ? 0.f : (v > 255.f ? 255.f : v));
		}
	}
}

/* Dense pyramidal Lucas-Kanade. Returns flow (u,v) such that
 * prev(x) ~= cur(x + (u,v)); i.e. cur(X) ~= prev(X - (u,v)). Frame is width x
 * height (each pyramid axis is halved independently). */
inline void denseFlowLK(const std::vector<float> &prev,
			const std::vector<float> &cur, int width, int height,
			std::vector<float> &u, std::vector<float> &v)
{
	const int LEVELS = 3, ITERS = 3, R = 5;
	const float LAMBDA = 1.0f, MAX_STEP = 2.0f;

	/* build prev/cur pyramids, coarsest last */
	std::vector<std::vector<float>> pp{prev}, cp{cur};
	std::vector<int> dimW{width}, dimH{height};
	for (int l = 1; l < LEVELS; ++l) {
		std::vector<float> dp, dc;
		int dw, dh;
		downsample2(pp.back(), dimW.back(), dimH.back(), dp, dw, dh);
		downsample2(cp.back(), dimW.back(), dimH.back(), dc, dw, dh);
		pp.push_back(std::move(dp));
		cp.push_back(std::move(dc));
		dimW.push_back(dw);
		dimH.push_back(dh);
	}

	int cw = dimW.back(), ch = dimH.back();
	u.assign((size_t)cw * ch, 0.f);
	v.assign((size_t)cw * ch, 0.f);

	for (int l = LEVELS - 1; l >= 0; --l) {
		const int w = dimW[l], h = dimH[l], n = w * h;
		if (w != cw || h != ch) { /* upsample flow from coarser level, scale x2 */
			std::vector<float> nu((size_t)n), nv((size_t)n);
			const float sx_ = (float)cw / (float)w; /* ~0.5 */
			const float sy_ = (float)ch / (float)h; /* ~0.5 */
			for (int y = 0; y < h; ++y)
				for (int x = 0; x < w; ++x) {
					float sx = x * sx_, sy = y * sy_;
					nu[(size_t)y * w + x] = 2.f * bilinear(u, cw, ch, sx, sy);
					nv[(size_t)y * w + x] = 2.f * bilinear(v, cw, ch, sx, sy);
				}
			u.swap(nu);
			v.swap(nv);
			cw = w;
			ch = h;
		}

		/* cur gradients at this level (central diff) */
		const std::vector<float> &cl_ = cp[l];
		const std::vector<float> &pl_ = pp[l];
		std::vector<float> gx((size_t)n), gy((size_t)n);
		auto cix = [&](int x, int y) { return cl_[(size_t)y * w + x]; };
		for (int y = 0; y < h; ++y)
			for (int x = 0; x < w; ++x) {
				int xm = x > 0 ? x - 1 : 0, xp = x < w - 1 ? x + 1 : w - 1;
				int ym = y > 0 ? y - 1 : 0, yp = y < h - 1 ? y + 1 : h - 1;
				gx[(size_t)y * w + x] = 0.5f * (cix(xp, y) - cix(xm, y));
				gy[(size_t)y * w + x] = 0.5f * (cix(x, yp) - cix(x, ym));
			}

		for (int it = 0; it < ITERS; ++it) {
			std::vector<float> Gxx((size_t)n), Gyy((size_t)n),
				Gxy((size_t)n), Bx((size_t)n), By((size_t)n);
			for (int y = 0; y < h; ++y)
				for (int x = 0; x < w; ++x) {
					const size_t i = (size_t)y * w + x;
					const float px = x + u[i], py = y + v[i];
					const float Jw = bilinear(cl_, w, h, px, py);
					const float gxv = bilinear(gx, w, h, px, py);
					const float gyv = bilinear(gy, w, h, px, py);
					const float It = Jw - pl_[i];
					Gxx[i] = gxv * gxv;
					Gyy[i] = gyv * gyv;
					Gxy[i] = gxv * gyv;
					Bx[i] = gxv * It;
					By[i] = gyv * It;
				}
			boxSum(Gxx, w, h, R);
			boxSum(Gyy, w, h, R);
			boxSum(Gxy, w, h, R);
			boxSum(Bx, w, h, R);
			boxSum(By, w, h, R);
			for (size_t i = 0; i < (size_t)n; ++i) {
				const float a = Gxx[i] + LAMBDA, dd = Gyy[i] + LAMBDA,
					    b = Gxy[i];
				const float det = a * dd - b * b;
				if (det <= 1e-6f)
					continue;
				/* d = -H^-1 B */
				float du = -(dd * Bx[i] - b * By[i]) / det;
				float dv = -(a * By[i] - b * Bx[i]) / det;
				du = std::clamp(du, -MAX_STEP, MAX_STEP);
				dv = std::clamp(dv, -MAX_STEP, MAX_STEP);
				u[i] += du;
				v[i] += dv;
			}
		}
	}
}

} // namespace nr3d

enum class TemporalMode {
	Legacy = 0,
	HistoryClip = 1,
	ReactiveClip = 2,
};

class FlowStabilizer {
public:
	/* rgba: w*h*4 tight (current frame, RGBA8).
	 * raw: w*h raw inverse depth (higher = nearer) from the model.
	 * out: w*h stabilised, normalised depth in [0,1].
	 * strength 0..1 (higher = steadier); smooth_sigma >=0 edge softening. */
	void process(const uint8_t *rgba, int w, int h, const std::vector<float> &raw,
		     std::vector<float> &out, bool enabled, TemporalMode mode,
		     float strength,
		     float smooth_sigma)
	{
		const int n = w * h;

		/* grayscale (Rec.601 luma, 0..255) */
		std::vector<float> gray((size_t)n);
		for (int i = 0; i < n; ++i)
			gray[i] = 0.299f * rgba[(size_t)i * 4 + 0] +
				  0.587f * rgba[(size_t)i * 4 + 1] +
				  0.114f * rgba[(size_t)i * 4 + 2];

		/* raw depth range */
		const float *d = raw.data();
		float mn = d[0], mx = d[0];
		for (int i = 1; i < n; ++i) {
			if (d[i] < mn)
				mn = d[i];
			if (d[i] > mx)
				mx = d[i];
		}

		/* scene-cut detection: mean abs luma diff vs previous frame */
		bool cut = false;
		if (has_prev_ && (int)prev_gray_.size() == n) {
			double s = 0.0;
			for (int i = 0; i < n; ++i)
				s += std::fabs(gray[i] - prev_gray_[i]);
			cut = (s / n) > CUT_THRESH;
		}

		/* normalisation scale: EMA while stabilising & continuous; snap to the
		 * current frame on a cut / first frame / when stabilisation is off */
		if (!enabled || !have_range_ || cut) {
			mn_ema_ = mn;
			mx_ema_ = mx;
		} else {
			mn_ema_ = RANGE_EMA * mn_ema_ + (1.f - RANGE_EMA) * mn;
			mx_ema_ = RANGE_EMA * mx_ema_ + (1.f - RANGE_EMA) * mx;
		}
		have_range_ = true;
		const float inv = (mx_ema_ > mn_ema_) ? 1.f / (mx_ema_ - mn_ema_) : 0.f;

		std::vector<float> cur((size_t)n);
		for (int i = 0; i < n; ++i) {
			float v = (d[i] - mn_ema_) * inv; /* 1 = nearest */
			cur[i] = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
		}

		const bool do_blend = enabled && has_prev_ && !cut &&
				      (int)prev_depth_.size() == n &&
				      (int)prev_gray_.size() == n;
		if (!do_blend) {
			out = cur;
		} else {
			std::vector<float> u, vv;
			nr3d::denseFlowLK(prev_gray_, gray, w, h, u, vv);

			/* backward-warp prev depth AND prev gray into current alignment:
			 * warped(X) = prev(X - flow). The flow-compensated gray residual
			 * marks where the flow failed to explain the image motion (thin /
			 * fast movers the LK missed); there we trust the fresh depth even
			 * when the *depth* residual is small (low depth-contrast movers,
			 * e.g. a waved stick over a similar-depth background). */
			std::vector<float> warped((size_t)n), wgray((size_t)n);
			for (int y = 0; y < h; ++y)
				for (int x = 0; x < w; ++x) {
					const size_t i = (size_t)y * w + x;
					const float sx = x - u[i], sy = y - vv[i];
					warped[i] = nr3d::bilinear(prev_depth_, w, h, sx, sy);
					wgray[i] = nr3d::bilinear(prev_gray_, w, h, sx, sy);
				}

			const std::vector<float> *history = &warped;
			std::vector<float> clipped_history;
			if (mode != TemporalMode::Legacy) {
				nr3d::clipHistory3x3(cur, warped, w, h, clipped_history);
				history = &clipped_history;
			}

			std::vector<uint8_t> reactive;
			if (mode == TemporalMode::ReactiveClip) {
				reactive.assign((size_t)n, 0);
				for (int i = 0; i < n; ++i) {
					const float dres = std::fabs(cur[i] - warped[i]);
					const float ires = std::fabs(gray[i] - wgray[i]);
					const float direct =
						std::fabs(gray[i] - prev_gray_[i]);
					if (dres > REACTIVE_DEPTH ||
					    ires > REACTIVE_IMAGE ||
					    direct > REACTIVE_DIRECT)
						reactive[i] = 1;
				}
				nr3d::dilateMask(reactive, w, h, REACTIVE_DILATE);
			}

			float s = strength < 0.f ? 0.f
						 : (strength > 1.f ? 1.f : strength);
			const float alpha = 1.f - 0.92f * s;
			out.resize((size_t)n);
			for (int i = 0; i < n; ++i) {
				if (!reactive.empty() && reactive[i]) {
					out[i] = cur[i];
					continue;
				}
				float dres = std::fabs(cur[i] - warped[i]);  /* depth resid 0..1 */
				float dboost = (dres - GHOST_LO) * GHOST_INV_SPAN;
				float ires = std::fabs(gray[i] - wgray[i]);  /* image resid 0..255 */
				float iboost = (ires - IMG_LO) * IMG_INV_SPAN;
				float boost = dboost > iboost ? dboost : iboost;
				boost = boost < 0.f ? 0.f : (boost > 1.f ? 1.f : boost);
				float a = alpha + (1.f - alpha) * boost;
				out[i] = a * cur[i] + (1.f - a) * (*history)[i];
			}
		}

		prev_gray_ = gray;
		prev_depth_ = out; /* copy: baseline for next frame */
		has_prev_ = enabled; /* off -> fresh start when re-enabled */

		/* Soften depth edges so the backward warp feathers instead of tearing
		 * at silhouettes (DIBR rubber-sheet, grows with disparity). Edge-
		 * localised: only the depth discontinuities are feathered, so flat
		 * regions keep their full depth detail (a global blur rounded off the
		 * whole 3D and bled the foreground outward). */
		if (smooth_sigma > 0.f && (int)out.size() == n)
			nr3d::edgeSoftenDepth(out, w, h, smooth_sigma, EDGE_LO,
					      EDGE_HI);
	}

private:
	static constexpr float RANGE_EMA = 0.90f;       /* normalisation scale LPF */
	static constexpr float CUT_THRESH = 30.0f;      /* mean abs luma diff (0..255) */
	static constexpr float GHOST_LO = 0.25f;        /* anti-ghost depth-residual fade lo */
	static constexpr float GHOST_INV_SPAN = 1.0f / 0.25f;
	static constexpr float IMG_LO = 18.0f;          /* anti-ghost image-residual fade lo (luma 0..255) */
	static constexpr float IMG_INV_SPAN = 1.0f / 40.0f;
	static constexpr float REACTIVE_DEPTH = 0.15f;  /* discard stale history */
	static constexpr float REACTIVE_IMAGE = 18.0f;  /* warped luma residual */
	static constexpr float REACTIVE_DIRECT = 30.0f; /* thin fast-motion fallback */
	static constexpr int REACTIVE_DILATE = 2;
	/* edge-softening: depth-gradient magnitude (per px, normalised depth) below
	 * EDGE_LO stays crisp, at/above EDGE_HI is fully feathered, smoothstep between */
	static constexpr float EDGE_LO = 0.05f;
	static constexpr float EDGE_HI = 0.25f;
	bool has_prev_ = false;
	bool have_range_ = false;
	float mn_ema_ = 0.f, mx_ema_ = 0.f;
	std::vector<float> prev_gray_, prev_depth_;
};
