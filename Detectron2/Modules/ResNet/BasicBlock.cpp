#include "Base.h"
#include "BasicBlock.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BasicBlockImpl::BasicBlockImpl(int in_channels, int out_channels, int stride, BatchNorm::Type norm) :
	CNNBlockBaseImpl(in_channels, out_channels, stride),
	m_convbn1(nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1).bias(false), norm),
	m_convbn2(nn::Conv2dOptions(out_channels, out_channels, 3).stride(stride).padding(1).bias(false), norm) {
	register_module("conv1", m_convbn1);
	register_module("conv2", m_convbn2);
	if (in_channels != out_channels) {
		m_shortcut = ConvBn2d(nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false), norm);
		register_module("shortcut", m_shortcut);
	}
}

void BasicBlockImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	if (m_shortcut) {
		m_shortcut->initialize(importer, prefix + ".shortcut", ModelImporter::kCaffe2MSRAFill);
	}
	m_convbn1->initialize(importer, prefix + ".conv1", ModelImporter::kCaffe2MSRAFill);
	m_convbn2->initialize(importer, prefix + ".conv2", ModelImporter::kCaffe2MSRAFill);
}

torch::Tensor BasicBlockImpl::forward(torch::Tensor x) {
	auto out = m_convbn1(x);
	out = relu(out);
	out = m_convbn2(out);

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
