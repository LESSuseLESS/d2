#include "Base.h"
#include "Res5ROIHeads.h"

#include <Detectron2/Modules/BatchNorm/BatchNorm.h>
#include <Detectron2/Modules/ResNet/BottleneckBlock.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Res5ROIHeadsImpl::Res5ROIHeadsImpl(CfgNode &cfg, const ShapeSpec::Map &input_shapes) : ROIHeadsImpl(cfg),
	m_in_features(cfg["MODEL.ROI_HEADS.IN_FEATURES"].as<vector<string>>()),
	m_mask_on(cfg["MODEL.MASK_ON"].as<bool>())
{
	auto pooler_resolution = cfg["MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION"].as<int>();
	auto pooler_type = cfg["MODEL.ROI_BOX_HEAD.POOLER_TYPE"].as<string>();
	auto sampling_ratio = cfg["MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO"].as<int>();

	assert(m_in_features.size() == 1);
	auto iter = input_shapes.find(m_in_features[0]);
	assert(iter != input_shapes.end());

	float pooler_scales = 1.0f / iter->second.stride;
	m_pooler = shared_ptr<ROIPoolerImpl>(
		new ROIPoolerImpl(pooler_type, { pooler_resolution, pooler_resolution }, { pooler_scales }, sampling_ratio));
	register_module("pooler", m_pooler);

	int out_channels = _build_res5_block(cfg);
	register_module("res5", m_res5);
	{
		ShapeSpec shape{};
		shape.channels = out_channels;
		shape.height = shape.width = 1;
		m_box_predictor = shared_ptr<FastRCNNOutputLayersImpl>(new FastRCNNOutputLayersImpl(cfg, shape));
		register_module("box_predictor", m_box_predictor);
	}
	if (m_mask_on) {
		ShapeSpec shape{};
		shape.channels = out_channels;
		shape.height = shape.width = pooler_resolution;
		m_mask_head = build_mask_head(cfg, shape);
		register_module("mask_head", m_mask_head);
	}
}

int Res5ROIHeadsImpl::_build_res5_block(CfgNode &cfg) {
	int stage_channel_factor = 2 * 2 * 2;			// res5 is 8x res2
	int num_groups = cfg["MODEL.RESNETS.NUM_GROUPS"].as<int>();
	int width_per_group = cfg["MODEL.RESNETS.WIDTH_PER_GROUP"].as<int>();
	int bottleneck_channels = num_groups * width_per_group * stage_channel_factor;
	int out_channels = cfg["MODEL.RESNETS.RES2_OUT_CHANNELS"].as<int>() * stage_channel_factor;
	bool stride_in_1x1 = cfg["MODEL.RESNETS.STRIDE_IN_1X1"].as<bool>();
	BatchNorm::Type norm = BatchNorm::GetType(cfg["MODEL.RESNETS.NORM"].as<string>());

	// assert(cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1]); // Deformable conv is not yet supported in res5 head."
	int in_channels = out_channels / 2;
	for (int i = 0; i < 3; i++) {
		CNNBlockBase block = shared_ptr<CNNBlockBaseImpl>(new BottleneckBlockImpl(in_channels, out_channels,
			bottleneck_channels, i == 0 ? 2 : 1, num_groups, norm, stride_in_1x1));
		m_res5->push_back(block);
		in_channels = out_channels;
	}
	return out_channels;
}

Tensor Res5ROIHeadsImpl::_shared_roi_transform(const TensorVec &features, const BoxesList &boxes) {
	auto x = m_pooler->forward(features, boxes);
	return m_res5->forward(x);
}

TensorVec Res5ROIHeadsImpl::select_features(const TensorMap &features) {
	TensorVec selected;
	selected.reserve(m_in_features.size());
	for (auto &feature : m_in_features) {
		auto iter = features.find(feature);
		assert(iter != features.end());
		selected.push_back(iter->second);
	}
	return selected;
}	

void Res5ROIHeadsImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	if (m_box_predictor) {
		m_box_predictor->initialize(importer, prefix + ".box_predictor");
	}
	for (int i = 0; i < 3; i++) {
		dynamic_pointer_cast<CNNBlockBaseImpl>(m_res5[i])->initialize(importer, FormatString("res5.%d", i));
	}
	if (m_mask_head) {
		m_mask_head->initialize(importer, prefix + ".mask_head");
	}
}

std::tuple<InstancesList, TensorMap> Res5ROIHeadsImpl::forward(const ImageList &, const TensorMap &features,
	InstancesList &proposals_, const InstancesList &targets) {
	InstancesList proposals_with_gt;
	InstancesList *proposals = &proposals_;
	if (is_training()) {
		proposals_with_gt = label_and_sample_proposals(proposals_, targets);
		proposals = &proposals_with_gt;
	}

	BoxesList proposal_boxes = proposals->getTensorVec("proposal_boxes");

	auto selected = select_features(features);
	auto box_features = _shared_roi_transform(selected, proposal_boxes);
	auto predictions = m_box_predictor->forward(box_features.mean({ 2, 3 }));

	if (is_training()) {
		auto losses = m_box_predictor->losses(predictions, *proposals);
		if (m_mask_on) {
			TensorVec fg_selection_masks;
			tie(*proposals, fg_selection_masks) = select_foreground_proposals(*proposals, m_num_classes);
			// Since the ROI feature transform is shared between boxes and masks,
			// we don't need to recompute features. The mask loss is only defined
			// on foreground proposals, so we need to select out the foreground
			// features.
			auto mask_features = box_features.index(torch::cat(fg_selection_masks, 0));
			update(losses, get<0>(m_mask_head(mask_features, *proposals)));
		}
		return { InstancesList{}, losses };
	}
	else {
		auto pred_instances = get<0>(m_box_predictor->inference(predictions, *proposals));
		pred_instances = forward_with_given_boxes(features, pred_instances);
		return { pred_instances, {} };
	}
}

InstancesList Res5ROIHeadsImpl::forward_with_given_boxes(const TensorMap &features, InstancesList &instances) {
	assert(!is_training());
	assert(!instances.empty() && instances[0]->has("pred_boxes") && instances[0]->has("pred_classes"));

	if (m_mask_on) {
		auto selected = select_features(features);
		auto x = _shared_roi_transform(selected, instances.getTensorVec("pred_boxes"));
		return get<1>(m_mask_head(x, instances));
	}

	return instances;
}
