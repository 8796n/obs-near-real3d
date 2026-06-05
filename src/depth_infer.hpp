// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 8796n <info@8796.jp>
/*
 * ONNX Runtime (DirectML EP) monocular-depth wrapper for obs-near-real3d.
 *
 * PC stand-in for libnr_mono_depth.so: runs Depth Anything V2 Small (exported to
 * ONNX, fixed square input) on the GPU via DirectML and returns a min/max
 * normalised relative-depth map (higher = nearer). DirectML needs only a DX12
 * GPU (no CUDA/cuDNN), which is why it is the standard choice for OBS plugins.
 */
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

class OrtDepth {
public:
	bool Init(const std::wstring &model_path)
	{
		try {
			Ort::SessionOptions so;
			so.SetIntraOpNumThreads(1);
			so.SetExecutionMode(ORT_SEQUENTIAL);
			so.DisableMemPattern();
			Ort::ThrowOnError(
				OrtSessionOptionsAppendExecutionProvider_DML(so, 0));
			session_ = Ort::Session(env_, model_path.c_str(), so);
			/* The exported model has a fixed input shape [1,3,H,W] (no dynamic
			 * axes -- DirectML is far slower with them), so read the inference
			 * dims straight from the model. A different export (e.g. a lighter
			 * 392x224 -lite model) then drives the whole pipeline with no code
			 * change: the caller sizes its textures/buffers from width()/height(). */
			std::vector<int64_t> shape =
				session_.GetInputTypeInfo(0)
					.GetTensorTypeAndShapeInfo()
					.GetShape();
			if (shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0) {
				last_error_ = "model input is not a fixed [1,3,H,W] shape";
				return false;
			}
			h_ = (int)shape[2];
			w_ = (int)shape[3];
			in_.resize((size_t)3 * w_ * h_);
			in_shape_ = {1, 3, h_, w_}; /* NCHW */
			return true;
		} catch (const std::exception &e) {
			last_error_ = e.what();
			return false;
		}
	}

	int width() const { return w_; }
	int height() const { return h_; }
	const std::string &last_error() const { return last_error_; }

	std::atomic<bool> temporal{true};      /* temporal stabilisation on/off */
	std::atomic<int> temporal_mode{0};     /* see TemporalMode in flow_stabilizer.hpp */
	std::atomic<float> stab_strength{0.4f}; /* 0 = none .. 1 = heavy smoothing */
	std::atomic<float> depth_smooth{0.3f}; /* 0..1 -> depth edge-softening sigma */

	/* rgba: w*h*4 tightly packed (R,G,B,A). out: w*h in [0,1]. */
	bool Run(const uint8_t *rgba, std::vector<float> &out)
	{
		static const float mean[3] = {0.485f, 0.456f, 0.406f};
		static const float istd[3] = {1.f / 0.229f, 1.f / 0.224f, 1.f / 0.225f};
		const int n = w_ * h_;
		float *r = in_.data(), *g = r + n, *b = g + n;
		for (int i = 0; i < n; ++i) {
			const uint8_t *p = rgba + (size_t)i * 4;
			r[i] = (p[0] / 255.f - mean[0]) * istd[0];
			g[i] = (p[1] / 255.f - mean[1]) * istd[1];
			b[i] = (p[2] / 255.f - mean[2]) * istd[2];
		}
		try {
			Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(
				OrtArenaAllocator, OrtMemTypeDefault);
			Ort::Value inv = Ort::Value::CreateTensor<float>(
				mi, in_.data(), in_.size(), in_shape_.data(), 4);
			const char *in_names[] = {"pixel_values"};
			const char *out_names[] = {"depth"};
			auto outs = session_.Run(Ort::RunOptions{nullptr}, in_names,
						 &inv, 1, out_names, 1);
			/* Validate the output before reading n floats: a different
			 * export / wrong model could return another dtype or element
			 * count and we'd otherwise read out of bounds. */
			auto ti = outs[0].GetTensorTypeAndShapeInfo();
			if (ti.GetElementType() !=
			    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
				last_error_ = "depth output is not float32";
				return false;
			}
			if (ti.GetElementCount() != (size_t)n) {
				last_error_ = "unexpected depth element count "
					      "(model output size != input WxH?)";
				return false;
			}
			/* raw inverse depth; normalisation + temporal stabilisation +
			 * scene-cut handling are all done downstream in FlowStabilizer
			 * (single place for temporal state). */
			const float *d = outs[0].GetTensorData<float>();
			out.assign(d, d + n);
			return true;
		} catch (const std::exception &e) {
			last_error_ = e.what();
			return false;
		}
	}

private:
	Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "obs-near-real3d"};
	Ort::Session session_{nullptr};
	int w_ = 0, h_ = 0;
	std::vector<float> in_;
	std::array<int64_t, 4> in_shape_{};
	std::string last_error_;
};
