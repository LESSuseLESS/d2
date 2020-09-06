#include "Base.h"
#include "Transform.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Transform::_set_attributes(const std::unordered_map<std::string, YAML::Node> &params) {
	for (auto iter : params) {
		auto &key = iter.first;
		if (key != "self" && key.find('_') != 0) {
			m_attrs[key] = iter.second;
		}
	}
}

torch::Tensor Transform::apply_box(torch::Tensor box) {
	// Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
	// ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
	Tensor idxs = torch::tensor(vector<int64_t>{ 0, 1, 2, 1, 0, 3, 2, 3 });
	auto coords = box.reshape({ -1, 4 }).index({ Colon, idxs }).reshape({ -1, 2 });
	coords = apply_coords(coords).reshape({ -1, 4, 2 });
	auto minxy = coords.min_values(1);
	auto maxxy = coords.max_values(1);
	auto trans_boxes = torch::cat({ minxy, maxxy }, 1);
	return trans_boxes;
}

TensorVec Transform::apply_polygons(const TensorVec &polygons) {
	TensorVec ret;
	ret.reserve(polygons.size());
	for (auto &p : polygons) {
		ret.push_back(apply_coords(p));
	}
	return ret;
}

std::shared_ptr<Transform> Transform::inverse() {
	assert(false);
	return nullptr;
}

std::string Transform::repr() const {
	assert(false);
	return "";
}

