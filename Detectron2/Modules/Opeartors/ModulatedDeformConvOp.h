#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from layers/deform_conv.py

	class _ModulatedDeformConv : public torch::autograd::Function<_ModulatedDeformConv> {
	public:
		static torch::autograd::variable_list forward(torch::autograd::AutogradContext *ctx,
			torch::Tensor input, torch::Tensor offset, torch::Tensor mask, torch::Tensor weight, torch::Tensor bias,
			int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1, int64_t groups = 1,
			int64_t deformable_groups = 1);

		static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
			torch::autograd::variable_list grad_output);

	private:
		static int _infer_shape(torch::autograd::AutogradContext *ctx, torch::Tensor input, torch::Tensor weight);
	};
	using modulated_deform_conv = _ModulatedDeformConv;
}