#include "Base.h"
#include "NewEmptyTensorOp.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

torch::autograd::variable_list _NewEmptyTensorOp::forward(torch::autograd::AutogradContext *ctx,
	const torch::Tensor &x, IntArrayRef new_shape) {
	ctx->saved_data["shape"] = x.sizes();
	return { x.new_empty(new_shape) };
}

torch::autograd::variable_list _NewEmptyTensorOp::backward(torch::autograd::AutogradContext *ctx,
	torch::autograd::variable_list grad) {
	auto shape = ctx->saved_data["shape"].toIntVector();
	return { grad[0].new_empty(shape), Tensor() };
}
