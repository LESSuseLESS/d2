#include "Base.h"
#include "DeformConvOp.h"

#include <Detectron2/detectron2/deformable/deform_conv.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

torch::autograd::variable_list _DeformConv::forward(torch::autograd::AutogradContext *ctx,
	torch::Tensor input, torch::Tensor offset, torch::Tensor weight,
	int64_t stride, int64_t padding, int64_t dilation,
	int64_t groups, int64_t deformable_groups, int64_t im2col_step) {
	// Expected 4D tensor as input, got {}D tensor instead.
	assert(input.numel() == 0 || input.dim() == 4);

	vector<int64_t> paddings{ padding, padding }, strides{ stride, stride }, dilations{ dilation, dilation };
	ctx->saved_data["stride"] = strides;
	ctx->saved_data["padding"] = paddings;
	ctx->saved_data["dilation"] = dilations;
	ctx->saved_data["groups"] = groups;
	ctx->saved_data["deformable_groups"] = deformable_groups;
	ctx->saved_data["im2col_step"] = im2col_step;

	ctx->save_for_backward({ input, offset, weight });

	auto output = input.new_empty(_output_size(input, weight, strides, paddings, dilations));

	Tensor empty0 = input.new_empty(0);
	Tensor empty1 = input.new_empty(0);
	ctx->saved_data["bufs_"] = TensorList{ empty0, empty1 };  // columns, ones

	assert(input.is_cuda()); // cpu version not implemented
	auto cur_im2col_step = _cal_im2col_step(input.size(0), im2col_step);
	assert(input.size(0) % cur_im2col_step == 0); // im2col step must divide batchsize

	detectron2::deform_conv_forward(input, weight, offset, output, empty0, empty1,
		weight.size(3), weight.size(2), stride, stride, padding, padding, dilation, dilation,
		groups, deformable_groups, cur_im2col_step);
	return { output };
}

torch::autograd::variable_list _DeformConv::backward(torch::autograd::AutogradContext *ctx,
	torch::autograd::variable_list grad_output) {
	auto saved = ctx->get_saved_variables();
	Tensor input = saved[0];
	Tensor offset = saved[1];
	Tensor weight = saved[2];

	Tensor grad_input, grad_offset, grad_weight;

	assert(grad_output[0].is_cuda()); // cpu version not implemented
	auto cur_im2col_step = _cal_im2col_step(input.size(0), ctx->saved_data["im2col_step"].to<int>());
	assert(input.size(0) % cur_im2col_step == 0); // im2col step must divide batchsize

	auto needs_input_grad = ctx->saved_data["needs_input_grad"].toTensor(); //!?
	if (needs_input_grad[0].item<bool>() || needs_input_grad[1].item<bool>()) {
		grad_input = torch::zeros_like(input);
		grad_offset = torch::zeros_like(offset);
		detectron2::deform_conv_backward_input(input, offset, grad_output[0], grad_input, grad_offset,
			weight,
			ctx->saved_data["bufs_"].toTensorList()[0],
			weight.size(3),
			weight.size(2),
			ctx->saved_data["stride"].toIntVector()[1],
			ctx->saved_data["stride"].toIntVector()[0],
			ctx->saved_data["padding"].toIntVector()[1],
			ctx->saved_data["padding"].toIntVector()[0],
			ctx->saved_data["dilation"].toIntVector()[1],
			ctx->saved_data["dilation"].toIntVector()[0],
			ctx->saved_data["groups"].toInt(),
			ctx->saved_data["deformable_groups"].toInt(),
			ctx->saved_data["cur_im2col_step"].toInt());
	}

	if (needs_input_grad[2].item<bool>()) {
		auto grad_weight = torch::zeros_like(weight);
		detectron2::deform_conv_backward_filter(input, offset, grad_output[0], grad_weight,
			ctx->saved_data["bufs_"].toTensorList()[0],
			ctx->saved_data["bufs_"].toTensorList()[1],
			weight.size(3),
			weight.size(2),
			ctx->saved_data["stride"].toIntVector()[1],
			ctx->saved_data["stride"].toIntVector()[0],
			ctx->saved_data["padding"].toIntVector()[1],
			ctx->saved_data["padding"].toIntVector()[0],
			ctx->saved_data["dilation"].toIntVector()[1],
			ctx->saved_data["dilation"].toIntVector()[0],
			ctx->saved_data["groups"].toInt(),
			ctx->saved_data["deformable_groups"].toInt(),
			1,
			ctx->saved_data["cur_im2col_step"].toInt());
	}
	ctx->mark_non_differentiable(grad_output); //!? @once_differentiable
	return { grad_input, grad_offset, grad_weight, Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
}

std::vector<int64_t> _DeformConv::_output_size(torch::Tensor input, torch::Tensor weight,
	const std::vector<int64_t> &stride, const std::vector<int64_t> &padding,
	const std::vector<int64_t> &dilation) {
	auto channels = weight.size(0);
	std::vector<int64_t> output_size{ input.size(0), channels };
	for (int d = 0; d < input.dim() - 2; d++) {
		auto in_size = input.size(d + 2);
		auto pad = padding[d];
		auto kernel = dilation[d] * (weight.size(d + 2) - 1) + 1;
		auto stride_ = stride[d];
		output_size.push_back((in_size + (2 * pad) - kernel) / stride_ + 1);
	}
	if (!all_vec<int64_t>(output_size, [](int64_t s){ return s > 0; })) {
		assert(false); // convolution input is too small
	}
	return output_size;
}

int64_t _DeformConv::_cal_im2col_step(int64_t input_size, int64_t default_size) {
	if (input_size <= default_size) {
		return input_size;
	}
	int64_t best_step = 1;
	auto max_step = min(int64_t(sqrt(input_size)) + 1, default_size);
	for (int64_t step = 2; step < max_step; step++) {
		if (input_size % step == 0) {
			if (input_size / step <= default_size) {
				return input_size / step;
			}
			best_step = step;
		}
	}
	return best_step;
}
