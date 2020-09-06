#include "Base.h"
#include "BasicStem.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BasicStemImpl::BasicStemImpl(int in_channels, int out_channels, BatchNorm::Type norm)
	: CNNBlockBaseImpl(in_channels, out_channels, 4), m_in_channels(in_channels),
	m_convbn1(nn::Conv2dOptions(in_channels, out_channels, 7).stride(2).padding(3).bias(false), norm) {
	register_module("conv1", m_convbn1);
}

void BasicStemImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	m_convbn1->initialize(importer, prefix + ".conv1", ModelImporter::kCaffe2MSRAFill);
}

torch::Tensor BasicStemImpl::forward(torch::Tensor x) {
	x = m_convbn1(x);
	x = relu_(x);
	x = max_pool2d(x, 3, 2, 1);
	return x;
}
