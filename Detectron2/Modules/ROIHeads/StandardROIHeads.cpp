#include "Base.h"
#include "StandardROIHeads.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

StandardROIHeadsImpl::StandardROIHeadsImpl(CfgNode &cfg) : ROIHeadsImpl(cfg),
	m_mask_on(false), m_keypoint_on(false)
{
}
void StandardROIHeadsImpl::Create(CfgNode &cfg, const ShapeSpec::Map &input_shapes) {
	m_mask_on = cfg["MODEL.MASK_ON"].as<bool>();
	m_keypoint_on = cfg["MODEL.KEYPOINT_ON"].as<bool>();
	m_train_on_pred_boxes = cfg["MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES"].as<bool>();

	// Subclasses that have not been updated to use from_config style construction
	// may have overridden _init_*_head methods. In this case, those overridden methods
	// will not be classmethods and we need to avoid trying to call them here.
	// We test for this with ismethod which only returns True for bound methods of cls.
	// Such subclasses will need to handle calling their overridden _init_*_head methods.
	_init_box_head(cfg, input_shapes);
	if (m_mask_on) _init_mask_head(cfg, input_shapes);
	if (m_keypoint_on) _init_keypoint_head(cfg, input_shapes);
}

int64_t StandardROIHeadsImpl::select_channels(const ShapeSpec::Map &input_shapes,
	const std::vector<std::string> &in_features) {
	int64_t selected = -1;
	for (auto &feature : in_features) {
		auto iter = input_shapes.find(feature);
		assert(iter != input_shapes.end());
		if (selected < 0) {
			selected = iter->second.channels;
		}
		else {
			assert(selected == iter->second.channels);
		}
	}
	return selected;
}

TensorVec StandardROIHeadsImpl::select_features(const TensorMap &features,
	const std::vector<std::string> &in_features) {
	TensorVec selected;
	selected.reserve(in_features.size());
	for (auto &feature : in_features) {
		auto iter = features.find(feature);
		assert(iter != features.end());
		selected.push_back(iter->second);
	}
	return selected;
}

std::vector<float> StandardROIHeadsImpl::get_pooler_scales(const ShapeSpec::Map &input_shapes,
	const std::vector<std::string> &in_features) {
	return vapply<float, string>(in_features,
		[=](string k) {
			auto iter = input_shapes.find(k);
			assert(iter != input_shapes.end());
			return (1.0 / iter->second.stride);
		});
}

void StandardROIHeadsImpl::_init_box_head(CfgNode &cfg, const ShapeSpec::Map &input_shapes) {
	m_box_in_features = cfg["MODEL.ROI_HEADS.IN_FEATURES"].as<vector<string>>();
	auto pooler_resolution = cfg["MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION"].as<int>();
	auto pooler_scales = get_pooler_scales(input_shapes, m_box_in_features);
	auto sampling_ratio = cfg["MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO"].as<int>();
	auto pooler_type = cfg["MODEL.ROI_BOX_HEAD.POOLER_TYPE"].as<string>();

	// If StandardROIHeads is applied on multiple feature maps (as in FPN),
	// then we share the same predictors and therefore the channel counts must be the same
	auto in_channels = select_channels(input_shapes, m_box_in_features);

	m_box_pooler = shared_ptr<ROIPoolerImpl>(new ROIPoolerImpl(pooler_type,
		{ pooler_resolution, pooler_resolution }, pooler_scales, sampling_ratio));

	// Here we split "box head" and "box predictor", which is mainly due to historical reasons.
	// They are used together so the "box predictor" layers should be part of the "box head".
	// New subclasses of ROIHeads do not need "box predictor"s.
	ShapeSpec shape;
	shape.channels = in_channels;
	shape.height = shape.width = pooler_resolution;
	m_box_heads = { build_box_head(cfg, shape) };
	register_module("box_head", m_box_heads[0]);
	m_box_predictors = { shared_ptr<FastRCNNOutputLayersImpl>(
		new FastRCNNOutputLayersImpl(cfg, m_box_heads[0]->output_shape())) };
	register_module("box_predictor", m_box_predictors[0]);
}

void StandardROIHeadsImpl::_init_mask_head(CfgNode &cfg, const ShapeSpec::Map &input_shapes) {
	m_mask_in_features = cfg["MODEL.ROI_HEADS.IN_FEATURES"].as<vector<string>>();
	auto pooler_resolution = cfg["MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION"].as<int>();
	auto pooler_scales = get_pooler_scales(input_shapes, m_mask_in_features);
	auto sampling_ratio = cfg["MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO"].as<int>();
	auto pooler_type = cfg["MODEL.ROI_MASK_HEAD.POOLER_TYPE"].as<string>();

	auto in_channels = select_channels(input_shapes, m_mask_in_features);

	m_mask_pooler = shared_ptr<ROIPoolerImpl>(new ROIPoolerImpl(pooler_type,
		{ pooler_resolution, pooler_resolution }, pooler_scales, sampling_ratio));
	register_module("mask_pooler", m_mask_pooler);

	ShapeSpec shape;
	shape.channels = in_channels;
	shape.height = shape.width = pooler_resolution;
	m_mask_head = build_mask_head(cfg, shape);
	register_module("mask_head", m_mask_head);
}

void StandardROIHeadsImpl::_init_keypoint_head(CfgNode &cfg, const ShapeSpec::Map &input_shapes) {
	m_keypoint_in_features = cfg["MODEL.ROI_HEADS.IN_FEATURES"].as<vector<string>>();
	auto pooler_resolution = cfg["MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION"].as<int>();
	auto pooler_scales = get_pooler_scales(input_shapes, m_keypoint_in_features);
	auto sampling_ratio = cfg["MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO"].as<int>();
	auto pooler_type = cfg["MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE"].as<string>();

	auto in_channels = select_channels(input_shapes, m_keypoint_in_features);

	m_keypoint_pooler = shared_ptr<ROIPoolerImpl>(new ROIPoolerImpl(pooler_type,
		{ pooler_resolution, pooler_resolution }, pooler_scales, sampling_ratio));
	register_module("keypoint_pooler", m_keypoint_pooler);

	ShapeSpec shape;
	shape.channels = in_channels;
	shape.height = shape.width = pooler_resolution;
	m_keypoint_head = build_keypoint_head(cfg, shape);
	register_module("keypoint_head", m_keypoint_head);
}

void StandardROIHeadsImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	if (m_box_heads.size() == 1) {
		m_box_heads[0]->initialize(importer, prefix + ".box_head");
	}
	else {
		for (int i = 0; i < m_box_heads.size(); i++) {
			m_box_heads[i]->initialize(importer, prefix + FormatString(".box_head.%d", i));
		}
	}
	if (m_box_predictors.size() == 1) {
		m_box_predictors[0]->initialize(importer, prefix + ".box_predictor");
	}
	else {
		for (int i = 0; i < m_box_predictors.size(); i++) {
			m_box_predictors[i]->initialize(importer, prefix + FormatString(".box_predictor.%d", i));
		}
	}
	if (m_mask_head) {
		m_mask_head->initialize(importer, prefix + ".mask_head");
	}
	if (m_keypoint_head) {
		m_keypoint_head->initialize(importer, prefix + ".keypoint_head");
	}
}

std::tuple<InstancesList, TensorMap> StandardROIHeadsImpl::forward(const ImageList &images, const TensorMap &features,
	InstancesList &proposals, const InstancesList &targets) {
	if (is_training()) {
		auto losses = get<0>(_forward_box(features, proposals));
		// Usually the original proposals used by the box head are used by the mask, keypoint
		// heads. But when `m_train_on_pred_boxes is True`, proposals will contain boxes
		// predicted by the box head.
		update(losses, get<0>(_forward_mask(features, proposals)));
		update(losses, get<0>(_forward_keypoint(features, proposals)));
		return { proposals, losses };
	}
	else {
		auto pred_instances = get<1>(_forward_box(features, proposals));
		// During inference cascaded prediction is used: the mask and keypoints heads are only
		// applied to the top scoring box detections.
		pred_instances = forward_with_given_boxes(features, pred_instances);
		return { pred_instances, {} };
	}
}

InstancesList StandardROIHeadsImpl::forward_with_given_boxes(const TensorMap &features, InstancesList &instances) {
	assert(!is_training());
	assert(!instances.empty() && instances[0]->has("pred_boxes") && instances[0]->has("pred_classes"));

	instances = get<1>(_forward_mask(features, instances));
	instances = get<1>(_forward_keypoint(features, instances));
	return instances;
}

std::tuple<TensorMap, InstancesList> StandardROIHeadsImpl::_forward_box(const TensorMap &features_,
	InstancesList &proposals) {
	auto features = select_features(features_, m_box_in_features);
	auto box_features = m_box_pooler(features, proposals.getTensorVec("proposal_boxes"));
	box_features = m_box_heads[0](box_features);
	auto predictions = m_box_predictors[0](box_features);

	if (is_training()) {
		auto losses = m_box_predictors[0]->losses(predictions, proposals);
		// proposals is modified in-place below, so losses must be computed first.
		if (m_train_on_pred_boxes) {
			torch::NoGradGuard guard;
			auto pred_boxes = m_box_predictors[0]->predict_boxes_for_gt_classes(
				predictions, proposals
			);
			int count = proposals.size();
			assert(pred_boxes.size() == count);
			for (int i = 0; i < count; i++) {
				auto proposals_per_image = proposals[i];
				auto pred_boxes_per_image = pred_boxes[i];
				proposals_per_image->set("proposal_boxes", pred_boxes_per_image);
			}
		}
		return { losses, {} };
	}
	else {
		auto pred_instances = get<0>(m_box_predictors[0]->inference(predictions, proposals));
		return { TensorMap{}, pred_instances };
	}
}

std::tuple<TensorMap, InstancesList> StandardROIHeadsImpl::_forward_mask(const TensorMap &features_,
	InstancesList &instances) {
	if (!m_mask_on) {
		if (is_training()) {
			return {};
		}
		else {
			return { TensorMap{}, instances };
		}
	}

	auto features = select_features(features_, m_mask_in_features);

	if (is_training()) {
		// The loss is only defined on positive proposals.
		auto proposals = get<0>(select_foreground_proposals(instances, m_num_classes));
		auto proposal_boxes = proposals.getTensorVec("proposal_boxes");
		auto mask_features = m_mask_pooler(features, proposal_boxes);
		return m_mask_head(mask_features, proposals);
	}
	else {
		auto pred_boxes = instances.getTensorVec("pred_boxes");
		auto mask_features = m_mask_pooler(features, pred_boxes);
		return m_mask_head(mask_features, instances);
	}
}

std::tuple<TensorMap, InstancesList> StandardROIHeadsImpl::_forward_keypoint(const TensorMap &features_,
	InstancesList &instances) {
	if (!m_keypoint_on) {
		if (is_training()) {
			return {};
		}
		else {
			return { TensorMap{}, instances };
		}
	}

	auto features = select_features(features_, m_keypoint_in_features);

	if (is_training()) {
		// The loss is defined on positive proposals with >=1 visible keypoints.
		auto proposals = get<0>(select_foreground_proposals(instances, m_num_classes));
		proposals = select_proposals_with_visible_keypoints(proposals);
		auto proposal_boxes = proposals.getTensorVec("proposal_boxes");

		auto keypoint_features = m_keypoint_pooler(features, proposal_boxes);
		return m_keypoint_head(keypoint_features, proposals);
	}
	else {
		auto pred_boxes = instances.getTensorVec("pred_boxes");
		auto keypoint_features = m_keypoint_pooler(features, pred_boxes);
		return m_keypoint_head(keypoint_features, instances);
	}
}
