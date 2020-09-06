#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from layers/wrappers.py

	class _NewEmptyTensorOp : public torch::autograd::Function<_NewEmptyTensorOp> {
	public:
		static torch::autograd::variable_list forward(torch::autograd::AutogradContext *ctx,
			const torch::Tensor &x, torch::IntArrayRef new_shape);

		static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
			torch::autograd::variable_list grad);
	};
}