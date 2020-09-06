#include "Base.h"
#include "BottleneckBlock.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BottleneckBlockImpl::BottleneckBlockImpl(int in_channels, int out_channels, int bottleneck_channels, int stride,
	int num_groups, BatchNorm::Type norm, bool stride_in_1x1, int dilation) :
	CNNBlockBaseImpl(in_channels, out_channels, stride),
	m_convbn1(nn::Conv2dOptions(in_channels, bottleneck_channels, 1).stride(stride_in_1x1 ? stride : 1)
		.bias(false), norm),
	m_convbn2(nn::Conv2dOptions(bottleneck_channels, bottleneck_channels, 3).stride(stride_in_1x1 ? 1 : stride)
		.padding(1 * dilation).bias(false).groups(num_groups).dilation(dilation), norm),
	m_convbn3(nn::Conv2dOptions(bottleneck_channels, out_channels, 1).bias(false), norm) {
	register_module("conv1", m_convbn1);
	register_module("conv2", m_convbn2);
	register_module("conv3", m_convbn3);
	// The original MSRA ResNet models have stride in the first 1x1 conv
	// The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have stride in the 3x3 conv
	// stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
	if (in_channels != out_channels) {
		m_shortcut = ConvBn2d(nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false), norm);
		register_module("shortcut", m_shortcut);
	}
}

void BottleneckBlockImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	if (m_shortcut) {
		m_shortcut->initialize(importer, prefix + ".shortcut", ModelImporter::kCaffe2MSRAFill);
	}
	m_convbn1->initialize(importer, prefix + ".conv1", ModelImporter::kCaffe2MSRAFill);
	m_convbn2->initialize(importer, prefix + ".conv2", ModelImporter::kCaffe2MSRAFill);
	m_convbn3->initialize(importer, prefix + ".conv3", ModelImporter::kCaffe2MSRAFill);

    // Zero-initialize the last normalization in each residual branch,
    // so that at the beginning, the residual branch starts with zeros,
    // and each residual block behaves like an identity.
    // See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
    // "For BN layers, the learnable scaling coefficient gamma is initialized
    // to be 1, except for each residual block's last BN
    // where gamma is initialized to be 0."

    // nn.init.constant_(self.conv3.norm.weight, 0)
    // TODO this somehow hurts performance when training GN models from scratch.
    // Add it as an option when we need to use this code to train a backbone.
}

torch::Tensor BottleneckBlockImpl::forward(torch::Tensor x) {
	auto out = m_convbn1(x);
	out = relu_(out);
	out = m_convbn2(out);
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
