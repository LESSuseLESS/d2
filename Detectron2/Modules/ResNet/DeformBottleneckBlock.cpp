#include "Base.h"
#include "DeformBottleneckBlock.h"

#include <Detectron2/Modules/Conv/DeformConv.h>
#include <Detectron2/Modules/Conv/ModulatedDeformConv.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DeformBottleneckBlockImpl::DeformBottleneckBlockImpl(int in_channels, int out_channels, int bottleneck_channels,
	int stride, int num_groups, BatchNorm::Type norm, bool stride_in_1x1, int dilation, bool deform_modulated,
	int deform_num_groups) :
	CNNBlockBaseImpl(in_channels, out_channels, stride), m_deform_modulated(deform_modulated),
	m_convbn1(nn::Conv2dOptions(in_channels, bottleneck_channels, 1).stride(stride_in_1x1 ? stride : 1)
		.bias(false), norm),
	m_convbn2_offset(nn::Conv2dOptions(bottleneck_channels, (deform_modulated ? 27 : 18) * deform_num_groups, 3)
		.stride(stride_in_1x1 ? 1 : stride).padding(1 * dilation).dilation(dilation), norm),
	m_convbn3(nn::Conv2dOptions(bottleneck_channels, out_channels, 1).bias(false), norm) {
	// offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size

	register_module("conv1", m_convbn1);
	register_module("conv2_offset", m_convbn2_offset);
	register_module("conv3", m_convbn3);

	if (in_channels != out_channels) {
		m_shortcut = ConvBn2d(nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false), norm);
		register_module("shortcut", m_shortcut);
	}

	int stride_3x3 = (stride_in_1x1 ? 1 : stride);
	assert(false);
	if (deform_modulated) {
		m_convbn2 = shared_ptr<Module>(new ModulatedDeformConvImpl(
			bottleneck_channels, bottleneck_channels, 3,
			stride_3x3, 1 * dilation, dilation, num_groups, deform_num_groups,
			false, norm));
	}
	else {
		m_convbn2 = shared_ptr<Module>(new DeformConvImpl(
			bottleneck_channels, bottleneck_channels, 3,
			stride_3x3, 1 * dilation, dilation, num_groups, deform_num_groups,
			false, norm));
	}
	register_module("conv2", m_convbn2);
}

void DeformBottleneckBlockImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	if (m_shortcut) {
		m_shortcut->initialize(importer, prefix + ".shortcut", ModelImporter::kCaffe2MSRAFill);
	}
	m_convbn1->initialize(importer, prefix + ".conv1", ModelImporter::kCaffe2MSRAFill);
	m_convbn2_offset->initialize(importer, prefix + ".conv2_offset", ModelImporter::kZeroFill);
	if (m_deform_modulated) {
		dynamic_cast<ModulatedDeformConvImpl*>(m_convbn2.get())->initialize(
			importer, prefix + ".conv2", ModelImporter::kCaffe2MSRAFill);
	}
	else {
		dynamic_cast<DeformConvImpl*>(m_convbn2.get())->initialize(
			importer, prefix + ".conv2", ModelImporter::kCaffe2MSRAFill);
	}
	m_convbn3->initialize(importer, prefix + ".conv3", ModelImporter::kCaffe2MSRAFill);
}

torch::Tensor DeformBottleneckBlockImpl::forward(torch::Tensor x) {
	auto out = m_convbn1(x);
	out = relu_(out);

	if (m_deform_modulated) {
		auto offset_mask = m_convbn2_offset(out);
		auto chunks = torch::chunk(offset_mask, 3, 1);
		auto offset_x = chunks[0];
		auto offset_y = chunks[1];
		auto mask = chunks[2];

		auto offset = torch::cat({ offset_x, offset_y }, 1);
		mask = mask.sigmoid();
		out = dynamic_cast<ModulatedDeformConvImpl*>(m_convbn2.get())->forward(out, offset, mask);
	}
	else {
		auto offset = m_convbn2_offset(out);
		out = dynamic_cast<DeformConvImpl*>(m_convbn2.get())->forward(out, offset);
	}
	out = relu_(out);

	out = m_convbn3(out);

	torch::Tensor shortcut;
	if (m_shortcut) {
		shortcut = m_shortcut(x);
	}
	else {
		shortcut = x;
	}

	out += shortcut;
	out = relu(out);
	return out;
}
