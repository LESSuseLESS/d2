#include "Base.h"
#include "RotatedAnchorGenerator.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RotatedAnchorGeneratorImpl::RotatedAnchorGeneratorImpl(CfgNode &cfg, const ShapeSpec::Vec &input_shapes) :
	RotatedAnchorGeneratorImpl(
		ShapeSpec::strides_vec(input_shapes),
		cfg["MODEL.ANCHOR_GENERATOR.SIZES"].as<vector<vector<float>>>(),
		cfg["MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS"].as<vector<vector<float>>>(),
		cfg["MODEL.ANCHOR_GENERATOR.ANGLES"].as<vector<vector<float>>>(),
		cfg["MODEL.ANCHOR_GENERATOR.OFFSET"].as<float>()) {
}

RotatedAnchorGeneratorImpl::RotatedAnchorGeneratorImpl(const vector<int> &strides,
	const vector<vector<float>> &sizes, const vector<vector<float>> &aspect_ratios,
	const vector<vector<float>> &angles, float offset) : AnchorGeneratorImpl(5),
	m_strides(strides),
	m_num_features(strides.size()),
	m_offset(offset)
{
	assert(0.0 <= m_offset && m_offset < 1.0);

	m_sizes = _broadcast_params(sizes, m_num_features);
	m_aspect_ratios = _broadcast_params(aspect_ratios, m_num_features);
	m_angles = _broadcast_params(angles, m_num_features);
	vector<int> counts = num_anchors();
	for (int i = 0; i < m_num_features; i++) {
		vector<float> anchors;
		anchors.reserve(counts[i] * 5);
		for (auto size : m_sizes[i]) {
			auto area = size * size;
			for (auto aspect_ratio : m_aspect_ratios[i]) {
				auto w = sqrt(area / aspect_ratio);
				auto h = aspect_ratio * w;
				for (auto a : m_angles[i]) {
					anchors.insert(anchors.end(), { 0.0f, 0.0f, w, h, a });
				}
			}
		}
		m_cell_anchors.push_back(torch::tensor(anchors).view({ -1, 5 }));
	}
	register_cell_anchors(m_cell_anchors);
}

vector<int> RotatedAnchorGeneratorImpl::num_anchors() const {
	vector<int> ret;
	for (int i = 0; i < m_num_features; i++) {
		ret.push_back(m_sizes[i].size() * m_aspect_ratios[i].size() * m_angles[i].size());
	}
	return ret;
}

void RotatedAnchorGeneratorImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	for (int i = 0; i < m_cell_anchors.size(); i++) {
		importer.Initialize(prefix + FormatString(".cell_anchors%d", i), m_cell_anchors[i]);
	}
}

BoxesList RotatedAnchorGeneratorImpl::forward(const TensorVec &features) {
	vector<pair<int, int>> grid_sizes;
	for (auto feature_map : features) {
		auto sizes = feature_map.sizes();
		int size = sizes.size();
		grid_sizes.push_back({ sizes[size - 2], sizes[size - 1] });
	}

	BoxesList anchors; // #featuremap tensors, each is (#locations x #cell_anchors) x 5
	for (int i = 0; i < grid_sizes.size(); i++) {
		auto &size = grid_sizes[i];
		auto &stride = m_strides[i];
		auto &base_anchors = m_cell_anchors[i];
		auto offsets = _create_grid_offsets(size, stride, m_offset, base_anchors.device());
		auto shift_x = offsets[0];
		auto shift_y = offsets[1];

		auto zeros = torch::zeros_like(shift_x);
		auto shifts = torch::stack({ shift_x, shift_y, zeros, zeros, zeros }, 1);

		anchors.push_back((shifts.view({ -1, 1, 5 }) + base_anchors.view({ 1, -1, 5 })).reshape({ -1, 5 }));
	}
	return anchors;
}
