#include "Base.h"
#include "ShapeSpec.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ShapeSpec::Vec ShapeSpec::filter(const Map &shapes, const std::vector<std::string> &names) {
	Vec filtered;
	filtered.reserve(names.size());
	for (auto name : names) {
		auto iter = shapes.find(name);
		assert(iter != shapes.end());
		auto &shape = iter->second;
		filtered.push_back(shape);
	}
	return filtered;
}

int ShapeSpec::channels_single(const Vec &shapes) {
	assert(!shapes.empty());
	int ret = shapes[0].channels;
	for (int i = 1; i < shapes.size(); i++) {
		assert(shapes[i].channels == ret);
	}
	return ret;
}

std::vector<int> ShapeSpec::channels_vec(const Vec &shapes) {
	vector<int> ret;
	ret.reserve(shapes.size());
	for (auto shape : shapes) {
		ret.push_back(shape.channels);
	}
	return ret;
}

std::vector<int> ShapeSpec::strides_vec(const ShapeSpec::Vec &shapes) {
	vector<int> ret;
	ret.reserve(shapes.size());
	for (auto shape : shapes) {
		ret.push_back(shape.stride);
	}
	return ret;
}
