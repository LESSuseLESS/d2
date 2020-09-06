#include "Base.h"
#include "ConvBn2d.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ConvBn2dImpl::ConvBn2dImpl(const torch::nn::Conv2dOptions &options, BatchNorm::Type norm, bool activation)
	: m_activation(activation) {
	m_conv = torch::nn::Conv2d(options);
	register_module("conv", m_conv);

	m_bn = BatchNorm(norm, options.out_channels());
	if (m_bn) {
		register_module("bn", m_bn.asModule());
	}
}

void ConvBn2dImpl::initialize(const ModelImporter &importer, const std::string &prefix, ModelImporter::Fill fill) {
	importer.Import(prefix, m_conv, fill);
	if (m_bn) {
		m_bn->initialize(importer, prefix + ".norm", fill);
	}
}

torch::Tensor ConvBn2dImpl::forward(torch::Tensor x) {
	x = m_conv(x);
	if (m_bn) {
		x = m_bn(x);
	}
	if (m_activation) {
		x = relu(x);
	}
	return x;
}
