#pragma once

#include "BatchNorm.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// This is to overwrite torch::nn::BatchNorm2dImpl and torch::nn::BatchNorm2d to have BatchNormImpl interface.

	class BatchNorm2dImpl : public torch::nn::BatchNorm2dImpl, public BatchNormImpl {
	public:
		BatchNorm2dImpl(const torch::nn::BatchNorm2dOptions &options) : torch::nn::BatchNorm2dImpl(options) {}

		// implementing BatchNormImpl
		virtual torch::Tensor &get_weight() override		{ return weight; }
		virtual torch::Tensor &get_bias() override			{ return bias; }
		virtual torch::Tensor *get_running_mean() override	{ return &running_mean; }
		virtual torch::Tensor *get_running_var() override	{ return &running_var; }
		virtual torch::Tensor forward(torch::Tensor x) override {
			return torch::nn::BatchNorm2dImpl::forward(x);
		}
	};
	TORCH_MODULE(BatchNorm2d);
}
