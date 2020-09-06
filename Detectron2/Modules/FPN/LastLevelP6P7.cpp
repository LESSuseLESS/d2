#include "Base.h"
#include "LastLevelP6P7.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

LastLevelP6P7Impl::LastLevelP6P7Impl(int64_t in_channels, int64_t out_channels, const char *in_feature) :
	m_p6(nn::Conv2dOptions(in_channels, out_channels, 3).stride(2).padding(1)),
	m_p7(nn::Conv2dOptions(out_channels, out_channels, 3).stride(2).padding(1))
{
	register_module("p6", m_p6);
	register_module("p7", m_p7);

	m_num_levels = 2;
	m_in_feature = in_feature;
}

void LastLevelP6P7Impl::initialize(const ModelImporter &importer, const std::string &prefix) {
	importer.Import(prefix + ".p6", m_p6, ModelImporter::kCaffe2XavierFill);
	importer.Import(prefix + ".p7", m_p7, ModelImporter::kCaffe2XavierFill);
}

TensorVec LastLevelP6P7Impl::forward(torch::Tensor c5) {
	auto x6 = m_p6(c5);
	auto x7 = m_p7(relu(x6));
	return { x6, x7 };
}
