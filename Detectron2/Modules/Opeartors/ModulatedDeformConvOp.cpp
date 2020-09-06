#include "Base.h"
#include "ModulatedDeformConvOp.h"

#include <Detectron2/detectron2/deformable/deform_conv.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

torch::autograd::variable_list _ModulatedDeformConv::forward(torch::autograd::AutogradContext *ctx,
	torch::Tensor input, torch::Tensor offset, torch::Tensor mask, torch::Tensor weight, torch::Tensor bias,
	int64_t stride, int64_t padding, int64_t dilation, int64_t groups, int64_t deformable_groups) {
	ctx->saved_data["stride"] = stride;
	ctx->saved_data["padding"] = padding;
	ctx->saved_data["dilation"] = dilation;
	ctx->saved_data["groups"] = groups;
	ctx->saved_data["deformable_groups"] = deformable_groups;
	bool with_bias = (bias.numel() != 0);
	if (!with_bias) {
		bias = input.new_empty(1);
	}
	ctx->saved_data["with_bias"] = with_bias;
	assert(input.is_cuda()); // cpu version not implemented

	if (weight.requires_grad() || mask.requires_grad() || offset.requires_grad() || input.requires_grad()) {
		ctx->save_for_backward({ input, offset, mask, weight, bias });
	}

	auto output = input.new_empty(_infer_shape(ctx, input, weight));

	Tensor empty0 = input.new_empty(0);
	Tensor empty1 = input.new_empty(0);
	ctx->saved_data["bufs_"] = TensorList{ empty0, empty1 };  // columns, ones

	detectron2::modulated_deform_conv_forward(input, weight, bias, empty0, offset, mask, output, empty1,
		weight.size(2), weight.size(3), stride, stride, padding, padding, dilation, dilation,
		groups, deformable_groups, with_bias);
	return { output };
}

torch::autograd::variable_list _ModulatedDeformConv::backward(torch::autograd::AutogradContext *ctx,
	torch::autograd::variable_list grad_output) {
	assert(grad_output[0].is_cuda()); // cpu version not implemented
	auto saved = ctx->get_saved_variables();
	Tensor input = saved[0];
	Tensor offset = saved[1];
	Tensor mask = saved[2];
	Tensor weight = saved[3];
	Tensor bias = saved[4];

	auto grad_input = torch::zeros_like(input);
	auto grad_offset = torch::zeros_like(offset);
	auto grad_mask = torch::zeros_like(mask);
	auto grad_weight = torch::zeros_like(weight);
	auto grad_bias = torch::zeros_like(bias);
	auto with_bias = ctx->saved_data["with_bias"].to<bool>();
	detectron2::modulated_deform_conv_backward(
		input,
		weight,
		bias,
		ctx->saved_data["bufs_"].toTensorList()[0],
		offset,
		mask,
		ctx->saved_data["bufs_"].toTensorList()[1],
		grad_input,
		grad_weight,
		grad_bias,
		grad_offset,
		grad_mask,
		grad_output[0],
		weight.size(2),
		weight.size(3),
		ctx->saved_data["stride"].toInt(),
		ctx->saved_data["stride"].toInt(),
		ctx->saved_data["padding"].toInt(),
		ctx->saved_data["padding"].toInt(),
		ctx->saved_data["dilation"].toInt(),
		ctx->saved_data["dilation"].toInt(),
		ctx->saved_data["groups"].toInt(),
		ctx->saved_data["deformable_groups"].toInt(),
		with_bias
	);
	if (!with_bias) {
		grad_bias.reset();
	}

	ctx->mark_non_differentiable(grad_output); //!? @once_differentiable
	return { grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
		Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
}

int _ModulatedDeformConv::_infer_shape(torch::autograd::AutogradContext *ctx, torch::Tensor input,
	torch::Tensor weight) {
	auto n = input.size(0);
	auto channels_out = weight.size(0);
	auto height = input.size(2);
	auto width = input.size(3);
	auto kernel_h = weight.size(2);
	auto kernel_w = weight.size(3);

	auto stride = ctx->saved_data["stride"].toInt();
	auto padding = ctx->saved_data["padding"].toInt();
	auto dilation = ctx->saved_data["dilation"].toInt();
	auto height_out = (height + 2 * padding - (dilation * (kernel_h - 1) + 1)) / stride + 1;
	auto width_out = (width + 2 * padding - (dilation * (kernel_w - 1) + 1)) / stride + 1;
	return n, channels_out, height_out, width_out;
}