#include "Base.h"
#include "TransformGen.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor TransformGen::_rand_range(double low, double *high, IntArrayRef size) {
	if (high == nullptr) {
		*high = low;
		low = 0;
	}
	return torch::rand(size).uniform_(low, *high);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TransformGen::TransformGen(const std::unordered_map<std::string, YAML::Node> &params) {
	for (auto iter : params) {
		auto &key = iter.first;
		if (key != "self" && key.find('_') != 0) {
			m_attrs[key] = iter.second;
		}
	}
}

std::string TransformGen::repr() const {
	assert(false);
	return "";
}
