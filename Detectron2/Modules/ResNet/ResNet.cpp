#include "Base.h"
#include "ResNet.h"

#include "BasicStem.h"
#include "BasicBlock.h"
#include "BottleneckBlock.h"
#include "DeformBottleneckBlock.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Backbone Detectron2::build_resnet_backbone(CfgNode &cfg, const ShapeSpec &input_shape) {
	// need registration of new blocks/stems?
	auto norm = BatchNorm::GetType(cfg["MODEL.RESNETS.NORM"].as<string>());
	BasicStem stem(
		input_shape.channels,
		cfg["MODEL.RESNETS.STEM_OUT_CHANNELS"].as<int>(),
		norm
	);

	auto freeze_at = cfg["MODEL.BACKBONE.FREEZE_AT"].as<int>();
	auto out_features_ = cfg["MODEL.RESNETS.OUT_FEATURES"].as<vector<string>>();
	unordered_set<string> out_features;
	out_features.insert(out_features_.begin(), out_features_.end());
	auto depth = cfg["MODEL.RESNETS.DEPTH"].as<int>();
	auto num_groups = cfg["MODEL.RESNETS.NUM_GROUPS"].as<int>();
	auto width_per_group = cfg["MODEL.RESNETS.WIDTH_PER_GROUP"].as<int>();
	int bottleneck_channels = num_groups * width_per_group;
	auto in_channels = cfg["MODEL.RESNETS.STEM_OUT_CHANNELS"].as<int>();
	auto out_channels = cfg["MODEL.RESNETS.RES2_OUT_CHANNELS"].as<int>();
	auto stride_in_1x1 = cfg["MODEL.RESNETS.STRIDE_IN_1X1"].as<bool>();
	auto res5_dilation = cfg["MODEL.RESNETS.RES5_DILATION"].as<int>();
	assert(res5_dilation == 1 || res5_dilation == 2);

	auto deform_on_per_stage = cfg["MODEL.RESNETS.DEFORM_ON_PER_STAGE"].as<vector<bool>>();
	auto deform_modulated = cfg["MODEL.RESNETS.DEFORM_MODULATED"].as<bool>();
	auto deform_num_groups = cfg["MODEL.RESNETS.DEFORM_NUM_GROUPS"].as<int>();

	auto num_blocks_per_stage = map<int, torch::IntList>{
		{ 18,  {2, 2, 2,  2}},
		{ 34,  {3, 4, 6,  3}},
		{ 50,  {3, 4, 6,  3}},
		{ 101, {3, 4, 23, 3}},
		{ 152, {3, 8, 36, 3}}
	}[depth];

	if (depth == 18 || depth == 34) {
		assert(out_channels == 64);
		for (auto deform : deform_on_per_stage) {
			assert(!deform); // MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34
		}
		assert(res5_dilation == 1);
		assert(num_groups == 1);
	}

	// Avoid creating variables without gradients
	// It consumes extra memory and may cause allreduce to fail
	int max_stage_idx = 0;
	for (auto f : out_features) {
		int idx = unordered_map<string, int>{ {"res2", 2}, {"res3", 3}, {"res4", 4}, {"res5", 5} }[f];
		if (idx > max_stage_idx) max_stage_idx = idx;
	}

	std::vector<std::vector<CNNBlockBase>> stages(max_stage_idx - 1);
	for (int stage_idx = 2; stage_idx < max_stage_idx + 1; stage_idx++) {
		int idx = stage_idx - 2;
		int dilation = (stage_idx == 5 ? res5_dilation : 1);
		int first_stride = ((idx == 0 || (stage_idx == 5 && dilation == 2)) ? 1 : 2);

		for (int i = 0; i < num_blocks_per_stage[idx]; i++) {
			int stride = (i == 0 ? first_stride : 1);
			shared_ptr<CNNBlockBaseImpl> block;
			if (depth == 18 || depth == 34) {
				// Use BasicBlock for R18 and R34.
				block = make_shared<BasicBlockImpl>(in_channels, out_channels, stride, norm);
			}
			else {
				if (deform_on_per_stage[idx]) {
					block = make_shared<DeformBottleneckBlockImpl>(in_channels, out_channels,
						bottleneck_channels, stride, num_groups, norm, stride_in_1x1, dilation, deform_modulated,
						deform_num_groups);
				}
				else {
					block = make_shared<BottleneckBlockImpl>(in_channels, out_channels,
						bottleneck_channels, stride, num_groups, norm, stride_in_1x1, dilation);
				}
			}
			stages[idx].push_back(block);
			in_channels = out_channels;
		}
		in_channels = out_channels;
		out_channels *= 2;
		bottleneck_channels *= 2;
	}

	auto ret = make_shared<ResNetImpl>(shared_ptr<CNNBlockBaseImpl>(stem.ptr()), stages, out_features);
	ret->freeze(freeze_at);
	return shared_ptr<BackboneImpl>(ret);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ResNetImpl::ResNetImpl(const CNNBlockBase &stem, const std::vector<std::vector<CNNBlockBase>> &stages,
	const std::unordered_set<std::string> &out_features, int num_classes) :
	m_stem(stem), m_out_features(out_features), m_num_classes(num_classes) {
	register_module("stem", m_stem);

	auto current_stride = m_stem->stride();

	auto &spec = m_output_shapes["stem"];
	spec.stride = current_stride;
	spec.channels = m_stem->out_channels();

	m_names.reserve(stages.size());
	m_stages.reserve(stages.size());
	int curr_channels;
	string name;
	for (int i = 0; i < stages.size(); i++) {
		auto stage = nn::Sequential();

		auto &blocks = stages[i];
		assert(!blocks.empty());
		for (int j = 0; j < blocks.size(); j++) {
			auto &block = blocks[j];
			current_stride *= block->stride();
			curr_channels = block->out_channels();
			stage->push_back(block);
		}

		name = FormatString("res%d", i + 2);
		m_stages.push_back(stage);
		register_module(name, stage);
		m_names.push_back(name);

		auto &spec = m_output_shapes[name];
		spec.stride = current_stride;
		spec.channels = curr_channels;
	}

	if (m_num_classes > 0) {
		m_avgpool = nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({ 1, 1 }));
		m_linear = nn::Linear(curr_channels, m_num_classes);
		register_module("linear", m_linear);
		name = "linear";
	}

	if (m_out_features.empty()) {
		m_out_features = { name };
	}
	unordered_set<string> children;
	for (auto iter : named_children()) {
		children.insert(iter.key());
	}
	for (auto out_feature : m_out_features) {
		assert(children.find(out_feature) != children.end());
	}
}

void ResNetImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	m_stem->initialize(importer, prefix + ".stem");

	for (int stageIndex = 0; stageIndex < m_stages.size(); stageIndex++) {
		auto &stage = m_stages[stageIndex];
		int blockIndex = 0;
		for (auto &block : stage->children()) {
			block->as<CNNBlockBaseImpl>()->initialize(importer,
				prefix + FormatString(".res%d", stageIndex + 2) + FormatString(".%d", blockIndex++));
		}
	}

	if (m_linear) {
		// Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
		// "The 1000-way fully-connected layer is initialized by
		// drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
		importer.Import(prefix + ".linear", m_linear, ModelImporter::kNormalFill2);
	}
}

TensorMap ResNetImpl::forward(torch::Tensor x) {
	TensorMap outputs;
	x = m_stem->forward(x);
	if (m_out_features.find("stem") != m_out_features.end()) {
		outputs["stem"] = x;
	}
	for (int i = 0; i < m_names.size(); i++) {
		auto &stage = m_stages[i];
		auto &name = m_names[i];
		x = stage->forward(x);
		if (m_out_features.find(name) != m_out_features.end()) {
			outputs[name] = x;
		}
	}
	if (m_linear) {
		x = m_avgpool(x);
		x = torch::flatten(x, 1);
		x = m_linear(x);
		if (m_out_features.find("linear") != m_out_features.end()) {
			outputs["linear"] = x;
		}
	}
	return outputs;
}

std::shared_ptr<ResNetImpl> ResNetImpl::freeze(int freeze_at) {
	if (freeze_at >= 1) {
		m_stem->freeze();
	}
	for (int i = 0; i < m_stages.size(); i++) {
		auto &stage = m_stages[i];
		if (freeze_at >= i + 2) {
			for (auto &block : stage->children()) {
				block->as<CNNBlockBaseImpl>()->freeze();
			}
		}
	}
	return dynamic_pointer_cast<ResNetImpl>(shared_from_this());
}
