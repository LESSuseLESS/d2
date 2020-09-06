#include "Base.h"
#include "LastLevelMaxPool.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

LastLevelMaxPoolImpl::LastLevelMaxPoolImpl() {
	m_num_levels = 1;
	m_in_feature = "p5";
}

void LastLevelMaxPoolImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	// do nothing
}

TensorVec LastLevelMaxPoolImpl::forward(torch::Tensor x) {
	torch::nn::functional::MaxPool2dFuncOptions options(1);
	return { torch::nn::functional::max_pool2d(x, options.stride(2).padding(0)) };
}
