#include "Base.h"
#include "DeformConv.h"

#include <Detectron2/Modules/Opeartors/DeformConvOp.h>
#include <Detectron2/Modules/Opeartors/NewEmptyTensorOp.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DeformConvImpl::DeformConvImpl(int in_channels, int out_channels, int kernel_size, int stride, int padding,
	int dilation, int groups, int deformable_groups, bool bias, BatchNorm::Type norm, bool activation) :
	m_in_channels(in_channels),
	m_out_channels(out_channels),
	m_kernel_size{ kernel_size, kernel_size },
	m_stride(stride),
	m_padding(padding),
	m_dilation(dilation),
	m_groups(groups),
	m_deformable_groups(deformable_groups),
	m_bias(bias),
	m_bn(norm, out_channels),
	m_activation(activation)
{
	assert(!bias);
	assert(in_channels % groups == 0);	// in_channels cannot be divisible by groups
	assert(out_channels % groups == 0); // out_channels cannot be divisible by groups

	m_weight = register_parameter("weight",
		torch::tensor({ out_channels, in_channels / m_groups, kernel_size, kernel_size }));
}

void DeformConvImpl::initialize(const ModelImporter &importer, const std::string &prefix, ModelImporter::Fill fill) {
	if (importer.HasData()) {
		importer.Initialize(prefix + ".weight", m_weight);
	}
	else {
		ModelImporter::FillTensor(m_weight, ModelImporter::kCaffe2MSRAFillIn);
	}
	if (m_bn) {
		m_bn->initialize(importer, prefix + ".norm", fill);
	}
}

torch::Tensor DeformConvImpl::forward(torch::Tensor x, torch::Tensor offset) {
	if (x.numel() == 0) {
		// When input is empty, we want to return a empty tensor with "correct" shape,
		// So that the following operations will not panic
		// if they check for the shape of the tensor.
		// This computes the height and width of the output tensor
		vector<int64_t> output_shape{ x.size(0), m_weight.size(0) };
		for (int d = 0; d < 2; d++) {
			auto i = x.size(-2 + d);
			auto p = m_padding;
			auto di = m_dilation;
			auto k = m_kernel_size[d];
			auto s = m_stride;

			output_shape.push_back((i + 2 * p - (di * (k - 1) + 1)) / s + 1);
		}
		return _NewEmptyTensorOp::apply(x, output_shape)[0];
	}

	x = deform_conv::apply(x, offset, m_weight, m_stride, m_padding, m_dilation, m_groups, m_deformable_groups)[0];
	if (m_bn) {
		x = m_bn(x);
	}
	if (m_activation) {
		x = relu(x);
	}
	return x;
}

std::string DeformConvImpl::extra_repr() const {
	std::string tmpstr;
	tmpstr = "in_channels=" + str(m_in_channels);
	tmpstr += ", out_channels=" + str(m_out_channels);
	tmpstr += ", kernel_size=" + str(m_kernel_size);
	tmpstr += ", stride=" + str(m_stride);
	tmpstr += ", padding=" + str(m_padding);
	tmpstr += ", dilation=" + str(m_dilation);
	tmpstr += ", groups=" + str(m_groups);
	tmpstr += ", deformable_groups=" + str(m_deformable_groups);
	tmpstr += ", bias=False";
	return tmpstr;
}
