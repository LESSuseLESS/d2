#include "Base.h"
#include "AnchorGenerator.h"

#include "DefaultAnchorGenerator.h"
#include "RotatedAnchorGenerator.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

AnchorGenerator Detectron2::build_anchor_generator(CfgNode &cfg, const ShapeSpec::Vec &input_shapes) {
	auto anchor_generator = cfg["MODEL.ANCHOR_GENERATOR.NAME"].as<string>();
	if (anchor_generator == "DefaultAnchorGenerator") {
		return shared_ptr<AnchorGeneratorImpl>(new DefaultAnchorGeneratorImpl(cfg, input_shapes));
	}
	if (anchor_generator == "RotatedAnchorGenerator") {
		return shared_ptr<AnchorGeneratorImpl>(new RotatedAnchorGeneratorImpl(cfg, input_shapes));
	}
	assert(false);
	return nullptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TensorVec AnchorGeneratorImpl::_create_grid_offsets(const pair<int, int> &size, int stride, float offset,
	torch::Device device) {
	int grid_height = size.first;
	int grid_width = size.second;

	auto options = TensorOptions(torch::kFloat32).device(device);
	auto shifts_x = torch::arange(offset * stride, grid_width * stride, stride, options);
	auto shifts_y = torch::arange(offset * stride, grid_height * stride, stride, options);
	auto vars = torch::meshgrid({ shifts_y, shifts_x });

	auto shift_y = vars[0];
	auto shift_x = vars[1];
	shift_x = shift_x.reshape(-1);
	shift_y = shift_y.reshape(-1);
	return { shift_x, shift_y };
}

vector<vector<float>> AnchorGeneratorImpl::_broadcast_params(const vector<vector<float>> &params, int num_features) {
	assert(!params.empty());
	if (params.size() == 1) {
		return vector<vector<float>>(num_features, params[0]);
	}
	assert(params.size() == num_features);
	return params;
}

vector<vector<float>> AnchorGeneratorImpl::_broadcast_params(const vector<float> &params, int num_features) {
	return vector<vector<float>>(num_features, params);
}

void AnchorGeneratorImpl::register_cell_anchors(const TensorVec &cell_anchors) {
	for (int i = 0; i < cell_anchors.size(); i++) {
		register_buffer(FormatString("%d", i), cell_anchors[i]);
	}
}
