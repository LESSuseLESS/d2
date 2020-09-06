#include "Base.h"
#include "FPN.h"

#include <Detectron2/Modules/ResNet/ResNet.h>

#include "LastLevelMaxPool.h"
#include "LastLevelP6P7.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Backbone Detectron2::build_backbone(CfgNode &cfg, const ShapeSpec *input_shape) {
	ShapeSpec default_input_shape;
	if (input_shape == nullptr) {
		default_input_shape.channels = cfg["MODEL.PIXEL_MEAN"].as<vector<float>>().size();
		input_shape = &default_input_shape;
	}
	auto backbone_name = cfg["MODEL.BACKBONE.NAME"].as<string>();
	if (backbone_name == "build_resnet_backbone") {
		return build_resnet_backbone(cfg, *input_shape);
	}
	if (backbone_name == "build_resnet_fpn_backbone") {
		return build_resnet_fpn_backbone(cfg, *input_shape);
	}
	if (backbone_name == "build_retinanet_resnet_fpn_backbone") {
		return build_retinanet_resnet_fpn_backbone(cfg, *input_shape);
	}
	assert(false);
	return nullptr;
}

Backbone Detectron2::build_resnet_fpn_backbone(CfgNode &cfg, const ShapeSpec &input_shape) {
	auto in_features = cfg["MODEL.FPN.IN_FEATURES"].as<vector<string>>();
	auto out_channels = cfg["MODEL.FPN.OUT_CHANNELS"].as<int64_t>();
	auto norm = BatchNorm::GetType(cfg["MODEL.FPN.NORM"].as<string>());
	auto fuse_type = cfg["MODEL.FPN.FUSE_TYPE"].as<string>();

	auto resnet = build_resnet_backbone(cfg, input_shape);

	// creating TopBlock
	shared_ptr<TopBlockImpl> topBlock = make_shared<LastLevelMaxPoolImpl>();

	// creating FPN
	return shared_ptr<BackboneImpl>(new FPNImpl(resnet, in_features, out_channels, norm, topBlock, fuse_type));
}

Backbone Detectron2::build_retinanet_resnet_fpn_backbone(CfgNode &cfg, const ShapeSpec &input_shape) {
	auto in_features = cfg["MODEL.FPN.IN_FEATURES"].as<vector<string>>();
	auto out_channels = cfg["MODEL.FPN.OUT_CHANNELS"].as<int64_t>();
	auto norm = BatchNorm::GetType(cfg["MODEL.FPN.NORM"].as<string>());
	auto fuse_type = cfg["MODEL.FPN.FUSE_TYPE"].as<string>();

	auto resnet = build_resnet_backbone(cfg, input_shape);

	// creating TopBlock
	auto &output_shapes = resnet->output_shapes();
	const char *in_feature = "res5";
	auto iter = output_shapes.find(in_feature);
	assert(iter != output_shapes.end());
	auto in_channels_p6p7 = iter->second.channels;
	shared_ptr<TopBlockImpl> topBlock = make_shared<LastLevelP6P7Impl>(in_channels_p6p7, out_channels, in_feature);

	// creating FPN
	return shared_ptr<BackboneImpl>(new FPNImpl(resnet, in_features, out_channels, norm, topBlock, fuse_type));
}

void FPNImpl::_assert_strides_are_log2_contiguous(const std::vector<int64_t> &strides) {
	for (int i = 1; i < strides.size(); i++) {
		auto stride = strides[i];
		assert(stride == 2 * strides[i - 1]);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FPNImpl::FPNImpl(Backbone bottom_up, const vector<string> &in_features, int out_channels, BatchNorm::Type norm,
	TopBlock top_block, const string &fuse_type) :
	m_bottom_up(bottom_up),
	m_in_features(in_features),
	m_top_block(top_block),
	m_fuse_type(fuse_type)
{
	assert(m_fuse_type == "avg" || m_fuse_type == "sum");

	register_module("bottom_up", m_bottom_up);
	register_module("top_block", m_top_block);

	const ShapeSpec::Map &input_shapes = bottom_up->output_shapes();
	vector<int64_t> in_channels, in_strides;
	in_channels.reserve(in_features.size());
	in_strides.reserve(in_features.size());
	for (auto &f : in_features) {
		auto iter = input_shapes.find(f);
		assert(iter != input_shapes.end());
		auto &item = iter->second;
		in_channels.push_back(item.channels);
		in_strides.push_back(item.stride);
	}

	_assert_strides_are_log2_contiguous(in_strides);
	m_stages.reserve(in_strides.size());
	for (int i = 0; i < in_strides.size(); i++) {
		m_stages.push_back(IntLog2(in_strides[i]));
	}

	bool bias = (norm == BatchNorm::kNone);
	m_lateral_convs.reserve(in_channels.size());
	m_output_convs.reserve(in_channels.size());
	for (int i = 0; i < in_channels.size(); i++) {
		ConvBn2d lateral_conv(nn::Conv2dOptions(in_channels[i], out_channels, 1).bias(bias), norm);
		ConvBn2d output_conv(nn::Conv2dOptions(out_channels, out_channels, 3).padding(1).bias(bias), norm);

		auto stage = m_stages[i];
		register_module(FormatString("fpn_lateral%d", stage), lateral_conv);
		register_module(FormatString("fpn_output%d", stage), output_conv);

		m_lateral_convs.push_back(lateral_conv);
		m_output_convs.push_back(output_conv);
	}

	// Place convs into top-down order (from low to high resolution) to make the top-down computation
	// in forward clearer.
	std::reverse(m_lateral_convs.begin(), m_lateral_convs.end());
	std::reverse(m_output_convs.begin(), m_output_convs.end());

	// Return feature names are "p<stage>", like["p2", "p3", ..., "p6"]
	int index = 0;
	for (int i = 0; i < in_strides.size(); i++) {
		auto stage = m_stages[i];
		auto name = FormatString("p%d", stage);
		auto &shape = m_output_shapes[name];

		shape.stride = in_strides[i];
		shape.channels = out_channels;
		shape.index = index++;
	}
	if (top_block) { // top block output feature maps
		auto stageLast = m_stages[in_strides.size() - 1];
		for (int level = 0; level < top_block->num_levels(); level++) {
			int stage = (stageLast + 1) + level;
			auto name = FormatString("p%d", stage);
			auto &shape = m_output_shapes[name];
			shape.stride = IntExp2(stage);
			shape.channels = out_channels;
			shape.index = index++;
		}
	}

	m_size_divisibility = in_strides[in_strides.size() - 1];
}

void FPNImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	m_bottom_up->initialize(importer, prefix + ".bottom_up");
	m_top_block->initialize(importer, prefix + ".top_block");

	int index = (m_lateral_convs.size() - 1);
	for (int i = 0; i < m_stages.size(); i++, --index) {
		int stage = m_stages[i];
		m_lateral_convs[index]->initialize(importer, prefix + FormatString(".fpn_lateral%d", stage),
			ModelImporter::kNoFill);
		m_output_convs[index]->initialize(importer, prefix + FormatString(".fpn_output%d", stage),
			ModelImporter::kNoFill);
	}
}

TensorMap FPNImpl::forward(torch::Tensor x) {
	TensorMap bottom_up_features = m_bottom_up->forward(x);

	// Reverse feature maps into top-down order (from low to high resolution)
	TensorVec features;
	features.reserve(m_in_features.size());
	for (int i = m_in_features.size() - 1; i >= 0; i--) {
		auto &name = m_in_features[i];
		auto iter = bottom_up_features.find(name);
		assert(iter != bottom_up_features.end());
		features.push_back(iter->second);
	}

	TensorVec results;
	results.reserve(m_in_features.size());
	{
		auto options = torch::nn::functional::InterpolateFuncOptions()
			.scale_factor(std::vector<double>({2, 2}))
			.mode(torch::kNearest);
		auto prev_features = m_lateral_convs[0]->forward(features[0]);
		results.push_back(m_output_convs[0]->forward(prev_features));
		for (int i = 1; i < features.size(); i++) {
			auto top_down_features = torch::nn::functional::interpolate(prev_features, options);
			auto lateral_features = m_lateral_convs[i]->forward(features[i]);
			prev_features = lateral_features + top_down_features;
			if (m_fuse_type == "avg") prev_features /= 2;
			results.push_back(m_output_convs[i]->forward(prev_features));
		}
		std::reverse(results.begin(), results.end());
	}

	if (m_top_block) {
		string in_feature = m_top_block->in_feature();
		torch::Tensor top_block_in_feature;
		auto iter = bottom_up_features.find(in_feature);
		if (iter != bottom_up_features.end()) {
			top_block_in_feature = iter->second;
		}
		else {
			auto iter = m_output_shapes.find(in_feature);
			assert(iter != m_output_shapes.end());
			int index = iter->second.index;
			assert(index >= 0 && index < results.size());
			top_block_in_feature = results[index];
		}
		TensorVec top_results = m_top_block->forward(top_block_in_feature);
		results.insert(results.end(), top_results.begin(), top_results.end());
	}
	assert(results.size() == m_output_shapes.size());
	TensorMap ret;
	for (auto iter : m_output_shapes) {
		ret[iter.first] = results[iter.second.index];
	}
	return ret;
}
