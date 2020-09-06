#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from layers/deform_conv.py

	class _DeformConv : public torch::autograd::Function<_DeformConv> {
	public:
		static torch::autograd::variable_list forward(torch::autograd::AutogradContext *ctx,
			torch::Tensor input, torch::Tensor offset, torch::Tensor weight,
			int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1, int64_t groups = 1,
			int64_t deformable_groups = 1, int64_t im2col_step = 64);

		//! @once_differentiable
		static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
			torch::autograd::variable_list grad_output);

	private:
		static std::vector<int64_t> _output_size(torch::Tensor input, torch::Tensor weight,
			const std::vector<int64_t> &stride, const std::vector<int64_t> &padding,
			const std::vector<int64_t> &dilation);

		//! @lru_cache(maxsize=128)
		/**
			Calculate proper im2col step size, which should be divisible by input_size and not larger
			than prefer_size. Meanwhile the step size should be as large as possible to be more
			efficient. So we choose the largest one among all divisors of input_size which are smaller
			than prefer_size.
			:param input_size: input batch size .
			:param default_size: default preferred im2col step size.
			:return: the largest proper step size.
		*/
		static int64_t _cal_im2col_step(int64_t input_size, int64_t default_size);
	};
	using deform_conv = _DeformConv;
}