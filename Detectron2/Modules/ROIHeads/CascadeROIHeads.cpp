#include "Base.h"
#include "CascadeROIHeads.h"

#include <Detectron2/Utils/EventStorage.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class _ScaleGradientOp : public torch::autograd::Function<_ScaleGradientOp> {
public:
	static torch::autograd::variable_list forward(torch::autograd::AutogradContext *ctx,
		const torch::Tensor &input, float scale) {
		ctx->saved_data["scale"] = scale;
		return { input };
	}

	static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
		torch::autograd::variable_list grad) {
		auto scale = ctx->saved_data["scale"].to<float>();
		return { grad[0] * scale, Tensor() };
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CascadeROIHeadsImpl::CascadeROIHeadsImpl(CfgNode &cfg, std::vector<BoxHead> box_heads,
	std::vector<FastRCNNOutputLayers> box_predictors, std::vector<std::shared_ptr<Matcher>> proposal_matchers) :
	StandardROIHeadsImpl(cfg)
{
	// The first matcher matches RPN proposals with ground truth, done in the base class
	m_num_cascade_stages = box_heads.size();
	assert(box_predictors.size() == m_num_cascade_stages);
	assert(proposal_matchers.size() == m_num_cascade_stages);

	m_box_heads = std::move(box_heads);
	m_box_predictors = std::move(box_predictors);
	m_proposal_matchers = std::move(proposal_matchers);
}

void CascadeROIHeadsImpl::_init_box_head(CfgNode &cfg, const ShapeSpec::Map &input_shapes) {
	m_box_in_features = cfg["MODEL.ROI_HEADS.IN_FEATURES"].as<vector<string>>();
	auto pooler_resolution = cfg["MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION"].as<int>();
	auto pooler_scales = get_pooler_scales(input_shapes, m_box_in_features);
	auto sampling_ratio = cfg["MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO"].as<int>();
	auto pooler_type = cfg["MODEL.ROI_BOX_HEAD.POOLER_TYPE"].as<string>();

	auto cascade_bbox_reg_weights = cfg["MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS"].as<vector<vector<float>>>();
	auto cascade_ious = cfg["MODEL.ROI_BOX_CASCADE_HEAD.IOUS"].as<vector<float>>();
	m_num_cascade_stages = cascade_ious.size();
	assert(cascade_bbox_reg_weights.size() == m_num_cascade_stages);
	// CascadeROIHeads only support class-agnostic regression now!
	assert(cfg["MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG"].as<bool>());
	assert(cascade_ious[0] == cfg["MODEL.ROI_HEADS.IOU_THRESHOLDS"].as<vector<float>>()[0]);

	// If StandardROIHeads is applied on multiple feature maps (as in FPN),
	// then we share the same predictors and therefore the channel counts must be the same
	auto in_channels = select_channels(input_shapes, m_box_in_features);

	m_box_pooler = shared_ptr<ROIPoolerImpl>(new ROIPoolerImpl(pooler_type,
		{ pooler_resolution, pooler_resolution }, pooler_scales, sampling_ratio));

	// Here we split "box head" and "box predictor", which is mainly due to historical reasons.
	// They are used together so the "box predictor" layers should be part of the "box head".
	// New subclasses of ROIHeads do not need "box predictor"s.
	ShapeSpec pooled_shape;
	pooled_shape.channels = in_channels;
	pooled_shape.height = pooled_shape.width = pooler_resolution;
	m_box_heads.reserve(m_num_cascade_stages);
	m_box_predictors.reserve(m_num_cascade_stages);
	m_proposal_matchers.reserve(m_num_cascade_stages);
	for (int i = 0; i < m_num_cascade_stages; i++) {
		auto match_iou = cascade_ious[i];
		auto bbox_reg_weights = cascade_bbox_reg_weights[i];

		auto box_head = build_box_head(cfg, pooled_shape);
		m_box_heads.push_back(box_head);

		auto bx = Box2BoxTransform::Create(bbox_reg_weights);
		auto out = make_shared<FastRCNNOutputLayersImpl>(cfg, box_head->output_shape(), bx);
		m_box_predictors.push_back(out);

		auto matcher = make_shared<Matcher>(vector<float>{ match_iou }, vector<int>{ 0, 1 }, false);
		m_proposal_matchers.push_back(matcher);
	}
}

std::tuple<InstancesList, TensorMap> CascadeROIHeadsImpl::forward(const ImageList &images, const TensorMap &features,
	InstancesList &proposals, const InstancesList &targets) {
	if (is_training()) {
		proposals = label_and_sample_proposals(proposals, targets);
	}

	if (is_training()) {
		// Need targets to box head
		auto losses = get<0>(_forward_box(features, proposals, targets));
		update(losses, get<0>(_forward_mask(features, proposals)));
		update(losses, get<0>(_forward_keypoint(features, proposals)));
		return { proposals, losses };
	}
	else {
		auto pred_instances = get<1>(_forward_box(features, proposals));
		pred_instances = forward_with_given_boxes(features, pred_instances);
		return { pred_instances, {} };
	}
}

std::tuple<TensorMap, InstancesList> CascadeROIHeadsImpl::_forward_box(const TensorMap &features_,
	InstancesList &proposals, const InstancesList &targets) {
	auto features = select_features(features_, m_box_in_features);

	// (predictor, predictions, proposals)
	vector<tuple<FastRCNNOutputLayers, TensorVec, InstancesList>> head_outputs;

	TensorVec prev_pred_boxes;
	auto image_sizes = proposals.getImageSizes();
	for (int k = 0; k < m_num_cascade_stages; k++) {
		if (k > 0) {
			// The output boxes of the previous stage are used to create the input
			// proposals of the next stage.
			proposals = _create_proposals_from_boxes(prev_pred_boxes, image_sizes);
			if (is_training()) {
				proposals = _match_and_label_boxes(proposals, k, targets);
			}
		}
		auto predictions = _run_stage(features, proposals, k);
		prev_pred_boxes = std::move(m_box_predictors[k]->predict_boxes(predictions, proposals));
		head_outputs.push_back({ m_box_predictors[k], std::move(predictions), std::move(proposals) });
	}

	if (is_training()) {
		TensorMap losses;
		auto &storage = get_event_storage();
		for (int stage = 0; stage < head_outputs.size(); stage++) {
			auto &predictor = get<0>(head_outputs[stage]);
			auto &predictions = get<1>(head_outputs[stage]);
			auto &proposals = get<2>(head_outputs[stage]);
			storage.push_name_scope(FormatString("stage%d", stage));
			auto stage_losses = predictor->losses(predictions, proposals);
			storage.pop_name_scope();
			for (auto iter : stage_losses) {
				string key = iter.first + FormatString("_stage%d", stage);
				losses[key] = iter.second;
			}
		}
		return { losses, {} };
	}
	else {
		// Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
		vector<TensorVec> scores_per_stage;
		scores_per_stage.reserve(head_outputs.size());
		for (auto &h : head_outputs) {
			auto scores = get<0>(h)->predict_probs(get<1>(h), get<2>(h));
			scores_per_stage.push_back(std::move(scores));
		}

		// Average the scores across heads
		TensorVec scores;
		for (int i = 0; i < scores_per_stage[0].size(); i++) {
			Tensor scores_per_image;
			for (int j = 0; j < scores_per_stage.size(); j++) {
				scores_per_image += scores_per_stage[j][i];
			}
			scores.push_back(scores_per_image * (1.0 / m_num_cascade_stages));
		}

		// Use the boxes of the last head
		auto &last = head_outputs[head_outputs.size() - 1];
		FastRCNNOutputLayers predictor = get<0>(last);
		TensorVec &predictions = get<1>(last);
		InstancesList &proposals = get<2>(last);
		auto boxes = predictor->predict_boxes(predictions, proposals);
		auto pred_instances = get<0>(fast_rcnn_inference(
			boxes,
			scores,
			image_sizes,
			predictor->m_test_score_thresh,
			predictor->m_test_nms_thresh,
			predictor->m_test_topk_per_image
		));
		return { TensorMap{}, pred_instances };
	}
}

InstancesList CascadeROIHeadsImpl::_match_and_label_boxes(InstancesList &proposals, int stage,
	const InstancesList &targets) {
	torch::NoGradGuard guard;

	int count = proposals.size();
	vector<int64_t> num_fg_samples; num_fg_samples.reserve(count);
	vector<int64_t> num_bg_samples; num_bg_samples.reserve(count);
	assert(targets.size() == count);
	for (int i = 0; i < count; i++) {
		auto proposals_per_image = proposals[i];
		auto targets_per_image = targets[i];
		auto match_quality_matrix = Boxes::pairwise_iou(
			targets_per_image->getTensor("gt_boxes"), proposals_per_image->getTensor("proposal_boxes")
		);
		// proposal_labels are 0 or 1
		Tensor matched_idxs, proposal_labels;
		tie(matched_idxs, proposal_labels) = (*m_proposal_matchers[stage])(match_quality_matrix);
		Tensor gt_classes, gt_boxes;
		if (targets_per_image->len() > 0) {
			gt_classes = targets_per_image->getTensor("gt_classes").index(matched_idxs);
			// Label unmatched proposals (0 label from matcher) as background (label=num_classes)
			gt_classes.index_put_({ proposal_labels == 0 }, m_num_classes);
			gt_boxes = targets_per_image->getTensor("gt_boxes").index(matched_idxs);
		}
		else {
			gt_classes = torch::zeros_like(matched_idxs) + m_num_classes;
			gt_boxes = Boxes::boxes(
				targets_per_image->getTensor("gt_boxes").new_zeros({ proposals_per_image->len(), 4 })
			)->tensor();
		}
		proposals_per_image->set("gt_classes", gt_classes);
		proposals_per_image->set("gt_boxes", gt_boxes);

		int num = (proposal_labels == 1).sum().item<int64_t>();
		num_bg_samples.push_back(num);
		num_fg_samples.push_back(proposal_labels.numel() - num);
	}

	// Log the number of fg/bg samples in each stage
	auto &storage = get_event_storage();
	storage.put_scalar(
		FormatString("stage%d/roi_head/num_fg_samples", stage),
		(float)(torch::tensor(num_fg_samples).sum().item<int64_t>()) / num_fg_samples.size());
	storage.put_scalar(
		FormatString("stage%d/roi_head/num_bg_samples", stage),
		(float)(torch::tensor(num_bg_samples).sum().item<int64_t>()) / num_bg_samples.size());
	return proposals;
}

TensorVec CascadeROIHeadsImpl::_run_stage(const TensorVec &features, InstancesList &proposals, int stage) {
	auto box_features = m_box_pooler(features, proposals.getTensorVec("proposal_boxes"));
	// The original implementation averages the losses among heads,
	// but scale up the parameter gradients of the heads.
	// This is equivalent to adding the losses among heads,
	// but scale down the gradients on features.
	box_features = _ScaleGradientOp::apply(box_features, 1.0f / m_num_cascade_stages)[0];
	box_features = m_box_heads[stage](box_features);
	return m_box_predictors[stage](box_features);
}

InstancesList CascadeROIHeadsImpl::_create_proposals_from_boxes(const TensorVec &boxes,
	const std::vector<ImageSize> &image_sizes) {
	// Just like RPN, the proposals should not have gradients
	int count = boxes.size();
	assert(image_sizes.size() == count);
	InstancesList proposals;
	proposals.reserve(count);
	for (int i = 0; i < count; i++) {
		auto boxes_per_image = boxes[i];
		auto &image_size = image_sizes[i];
		Boxes::boxes(boxes_per_image)->clip(image_size);
		if (is_training()) {
			// do not filter empty boxes at inference time,
			// because the scores from each stage need to be aligned and added later
			boxes_per_image = boxes_per_image.index(Boxes::boxes(boxes_per_image)->nonempty());
		}

		auto prop = make_shared<Instances>(image_size);
		prop->set("proposal_boxes", boxes_per_image);
		proposals.push_back(prop);
	}
	return proposals;
}
