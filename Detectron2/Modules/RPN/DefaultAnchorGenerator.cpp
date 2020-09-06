#include "Base.h"
#include "DefaultAnchorGenerator.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DefaultAnchorGeneratorImpl::DefaultAnchorGeneratorImpl(CfgNode &cfg, const ShapeSpec::Vec &input_shapes) :
	DefaultAnchorGeneratorImpl(
		ShapeSpec::strides_vec(input_shapes),
		cfg["MODEL.ANCHOR_GENERATOR.SIZES"].as<vector<vector<float>>>(),
		cfg["MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS"].as<vector<vector<float>>>(),
		cfg["MODEL.ANCHOR_GENERATOR.OFFSET"].as<float>()) {
}

DefaultAnchorGeneratorImpl::DefaultAnchorGeneratorImpl(const vector<int> &strides, const vector<vector<float>> &sizes,
	const vector<vector<float>> &aspect_ratios, float offset) : AnchorGeneratorImpl(4),
	m_strides(strides),
	m_num_features(strides.size()),
	m_offset(offset)
{
	assert(0.0 <= m_offset && m_offset < 1.0);

	// Generate a tensor storing canonical anchor boxes, which are all anchor boxes of different sizes and
	// aspect_ratios centered at (0, 0). We can later build the set of anchors for a full feature map by
	// shifting and tiling these tensors(see `meth:_grid_anchors`).
	//
	// This is different from the anchor generator defined in the original Faster R-CNN
	// code or Detectron. They yield the same AP, however the old version defines cell
	// anchors in a less natural way with a shift relative to the feature grid and
	// quantization that results in slightly different sizes for different aspect ratios.
	// See also https://github.com/facebookresearch/Detectron/issues/227

	m_sizes = _broadcast_params(sizes, m_num_features);
	m_aspect_ratios = _broadcast_params(aspect_ratios, m_num_features);
	vector<int> counts = num_anchors();
	for (int i = 0; i < m_num_features; i++) {
		vector<float> anchors;
		anchors.reserve(counts[i] * 4);
		for (auto size : m_sizes[i]) {
			auto area = size * size;
			for (auto aspect_ratio : m_aspect_ratios[i]) {
				auto w = sqrt(area / aspect_ratio);
				auto h = aspect_ratio * w;
				anchors.insert(anchors.end(), { -w / 2.0f, -h / 2.0f, w / 2.0f, h / 2.0f });
			}
		}
		m_cell_anchors.push_back(torch::tensor(anchors).view({-1, 4}));
	}
	register_cell_anchors(m_cell_anchors);
}

vector<int> DefaultAnchorGeneratorImpl::num_anchors() const {
	vector<int> ret;
	for (int i = 0; i < m_num_features; i++) {
		ret.push_back(m_sizes[i].size() * m_aspect_ratios[i].size());
	}
	return ret;
}

void DefaultAnchorGeneratorImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	for (int i = 0; i < m_cell_anchors.size(); i++) {
		importer.Initialize(prefix + FormatString(".cell_anchors.%d", i), m_cell_anchors[i]);
	}
}

BoxesList DefaultAnchorGeneratorImpl::forward(const TensorVec &features) {
	vector<pair<int, int>> grid_sizes;
	for (auto feature_map : features) {
		auto sizes = feature_map.sizes();
		int size = sizes.size();
		grid_sizes.push_back({ sizes[size - 2], sizes[size - 1] });
	}

	BoxesList anchors; // #featuremap tensors, each is (#locations x #cell_anchors) x 4
	for (int i = 0; i < grid_sizes.size(); i++) {
		auto &size = grid_sizes[i];
		auto &stride = m_strides[i];
		auto &base_anchors = m_cell_anchors[i];
		auto offsets = _create_grid_offsets(size, stride, m_offset, base_anchors.device());
		auto shift_x = offsets[0];
		auto shift_y = offsets[1];

		auto shifts = torch::stack({ shift_x, shift_y, shift_x, shift_y }, 1);
		anchors.push_back((shifts.view({ -1, 1, 4 }) + base_anchors.view({ 1, -1, 4 })).reshape({ -1, 4 }));
	}
	return anchors;
}
