#include "Base.h"
#include "RROIHeads.h"

#include <Detectron2/Utils/EventStorage.h>
#include <Detectron2/Structures/RotatedBoxes.h>
#include "RotatedFastRCNNOutputLayers.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RROIHeadsImpl::RROIHeadsImpl(CfgNode &cfg) : StandardROIHeadsImpl(cfg) {
}

void RROIHeadsImpl::Create(CfgNode &cfg, const ShapeSpec::Map &input_shapes) {
	StandardROIHeadsImpl::Create(cfg, input_shapes);
	assert(!m_mask_on && !m_keypoint_on); // Mask/Keypoints not supported in Rotated ROIHeads.
	assert(!m_train_on_pred_boxes); // train_on_pred_boxes not implemented for RROIHeads!
}

void RROIHeadsImpl::_init_box_head(CfgNode &cfg, const ShapeSpec::Map &input_shapes) {
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
	register_module("box_pooler", m_box_pooler);

	// Here we split "box head" and "box predictor", which is mainly due to historical reasons.
	// They are used together so the "box predictor" layers should be part of the "box head".
	// New subclasses of ROIHeads do not need "box predictor"s.
	ShapeSpec shape;
	shape.channels = in_channels;
	shape.height = shape.width = pooler_resolution;
	m_box_heads = { build_box_head(cfg, shape) };
	// This line is the only difference v.s. StandardROIHeads
	m_box_predictors = { shared_ptr<FastRCNNOutputLayersImpl>(
		new RotatedFastRCNNOutputLayersImpl(cfg, m_box_heads[0]->output_shape())) };
}

InstancesList RROIHeadsImpl::label_and_sample_proposals(InstancesList &proposals, const InstancesList &targets) {
	torch::NoGradGuard guard;

	auto gt_boxes = targets.getTensorVec("gt_boxes");
	if (m_proposal_append_gt) {
		add_ground_truth_to_proposals(proposals, gt_boxes);
	}

	int count = proposals.size();
	InstancesList proposals_with_gt; proposals_with_gt.reserve(count);
	vector<int64_t> num_fg_samples; num_fg_samples.reserve(count);
	vector<int64_t> num_bg_samples; num_bg_samples.reserve(count);
	assert(targets.size() == count);
	for (int i = 0; i < count; i++) {
		auto proposals_per_image = proposals[i];
		auto targets_per_image = targets[i];
		bool has_gt = targets_per_image->len() > 0;
		auto match_quality_matrix = RotatedBoxes::pairwise_iou_rotated(
			targets_per_image->getTensor("gt_boxes"), proposals_per_image->getTensor("proposal_boxes")
		);
		Tensor matched_idxs, matched_labels;
		tie(matched_idxs, matched_labels) = m_proposal_matcher(match_quality_matrix);
		Tensor sampled_idxs, gt_classes;
		tie(sampled_idxs, gt_classes) = _sample_proposals(
			matched_idxs, matched_labels, targets_per_image->getTensor("gt_classes")
		);

		proposals_per_image = (*proposals_per_image)[sampled_idxs];
		proposals_per_image->set("gt_classes", gt_classes);

		if (has_gt) {
			auto sampled_targets = matched_idxs.index(sampled_idxs);
			proposals_per_image->set("gt_boxes", targets_per_image->getTensor("gt_boxes").index(sampled_targets));
		}
		else {
			auto gt_boxes = RotatedBoxes(
				targets_per_image->getTensor("gt_boxes").new_zeros({ sampled_idxs.size(0), 5 })
			);
			proposals_per_image->set("gt_boxes", gt_boxes.tensor());
		}

		int num = (gt_classes == m_num_classes).sum().item<int64_t>();
		num_bg_samples.push_back(num);
		num_fg_samples.push_back(gt_classes.numel() - num);
		proposals_with_gt.push_back(proposals_per_image);
	}

    // Log the number of fg/bg samples that are selected for training ROI heads
	auto &storage = get_event_storage();
	storage.put_scalar("roi_head/num_fg_samples", torch::tensor(num_fg_samples).mean().item<float>());
	storage.put_scalar("roi_head/num_bg_samples", torch::tensor(num_bg_samples).mean().item<float>());

	return proposals_with_gt;
}
