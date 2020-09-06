#pragma once

#include "BatchNorm.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// This is to overwrite torch::nn::GroupNormImpl and torch::nn::GroupNorm to have BatchNormImpl interface.

	class GroupNormImpl : public torch::nn::GroupNormImpl, public BatchNormImpl {
	public:
		GroupNormImpl(const torch::nn::GroupNormOptions &options) : torch::nn::GroupNormImpl(options) {}

		// implementing BatchNormImpl
		virtual torch::Tensor &get_weight() override		{ return weight; }
		virtual torch::Tensor &get_bias() override			{ return bias; }
		virtual torch::Tensor *get_running_mean() override	{ return nullptr; }
		virtual torch::Tensor *get_running_var() override	{ return nullptr; }
		virtual torch::Tensor forward(torch::Tensor x) override {
			return torch::nn::GroupNormImpl::forward(x);
		}
	};
	TORCH_MODULE(GroupNorm);
}
