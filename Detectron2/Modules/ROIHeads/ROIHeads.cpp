#include "Base.h"
#include "RoiHeads.h"

#include <Detectron2/Utils/EventStorage.h>
#include <Detectron2/Structures/Sampling.h>
#include "CascadeROIHeads.h"
#include "Res5ROIHeads.h"
#include "RROIHeads.h"
#include "StandardROIHeads.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ROIHeads Detectron2::build_roi_heads(CfgNode &cfg, const ShapeSpec::Map &input_shapes) {
	auto name = cfg["MODEL.ROI_HEADS.NAME"].as<string>();
	if (name == "Res5ROIHeads") {
		return shared_ptr<ROIHeadsImpl>(new Res5ROIHeadsImpl(cfg, input_shapes));
	}
	if (name == "StandardROIHeads") {
		auto roiheads = make_shared<StandardROIHeadsImpl>(cfg);
		roiheads->Create(cfg, input_shapes);
		return shared_ptr<ROIHeadsImpl>(roiheads);
	}
	if (name == "RROIHeads") {
		auto roiheads = make_shared<RROIHeadsImpl>(cfg);
		roiheads->Create(cfg, input_shapes);
		return shared_ptr<ROIHeadsImpl>(roiheads);
	}
	if (name == "CascadeROIHeads") {
		auto roiheads = make_shared<CascadeROIHeadsImpl>(cfg);
		roiheads->Create(cfg, input_shapes);
		return shared_ptr<ROIHeadsImpl>(roiheads);
	}
	assert(false);
	return nullptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<InstancesList, TensorVec> ROIHeadsImpl::select_foreground_proposals(const InstancesList &proposals,
	int bg_label) {
	assert(!proposals.empty() && proposals[0]->has("gt_classes"));
	InstancesList fg_proposals; fg_proposals.reserve(proposals.size());
	TensorVec fg_selection_masks; fg_selection_masks.reserve(proposals.size());
	for (auto proposals_per_image : proposals) {
		auto gt_classes = proposals_per_image->getTensor("gt_classes");
		auto fg_selection_mask = (gt_classes != -1).bitwise_and(gt_classes != bg_label);
		auto fg_idxs = fg_selection_mask.nonzero().squeeze(1);
		fg_proposals.push_back((*proposals_per_image)[fg_idxs]);
		fg_selection_masks.push_back(fg_selection_mask);
	}
	return { fg_proposals, fg_selection_masks };
}

InstancesList ROIHeadsImpl::select_proposals_with_visible_keypoints(const InstancesList &proposals) {
	InstancesList ret; ret.reserve(proposals.size());
	vector<int64_t> all_num_fg; all_num_fg.reserve(proposals.size());
	for (auto proposals_per_image : proposals) {
		// If empty/unannotated image (hard negatives), skip filtering for train
		if (proposals_per_image->len() == 0) {
			ret.push_back(proposals_per_image);
			continue;
		}
		auto gt_keypoints = proposals_per_image->getTensor("gt_keypoints");
		// #fg x K x 3
		auto vis_mask = gt_keypoints.index({ Colon, Colon , 2 }) >= 1;
		auto xs = gt_keypoints.index({ Colon, Colon, 0 });
		auto ys = gt_keypoints.index({ Colon, Colon, 1 });
		auto proposal_boxes = proposals_per_image->getTensor("proposal_boxes").unsqueeze(1);  // #fg x 1 x 4
		auto kp_in_box = (
			(xs >= proposal_boxes.index({ Colon, Colon, 0 })).bitwise_and
			(xs <= proposal_boxes.index({ Colon, Colon, 2 })).bitwise_and
			(ys >= proposal_boxes.index({ Colon, Colon, 1 })).bitwise_and
			(ys <= proposal_boxes.index({ Colon, Colon, 3 }))
			);
		auto selection = (kp_in_box.bitwise_and(vis_mask)).any(1);
		auto selection_idxs = torch::nonzero(selection).index({ Colon, 0 });
		all_num_fg.push_back(selection_idxs.numel());
		ret.push_back((*proposals_per_image)[selection_idxs]);
	}

	auto &storage = get_event_storage();
	storage.put_scalar("keypoint_head/num_fg_samples", torch::tensor(all_num_fg).mean().item<float>());
	return ret;
}

void ROIHeadsImpl::update(TensorMap &dest, const TensorMap &src) {
	for (auto iter : src) {
		dest[iter.first] = iter.second;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ROIHeadsImpl::ROIHeadsImpl(CfgNode &cfg) :
	m_num_classes(cfg["MODEL.ROI_HEADS.NUM_CLASSES"].as<int>()),
	m_batch_size_per_image(cfg["MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE"].as<int>()),
	m_positive_sample_fraction(cfg["MODEL.ROI_HEADS.POSITIVE_FRACTION"].as<float>()),
	m_proposal_matcher(
		cfg["MODEL.ROI_HEADS.IOU_THRESHOLDS"].as<vector<float>>(),
		cfg["MODEL.ROI_HEADS.IOU_LABELS"].as<vector<int>>(),
		false // allow_low_quality_matches
	),
	m_proposal_append_gt(cfg["MODEL.ROI_HEADS.PROPOSAL_APPEND_GT"].as<bool>())
{
}

std::tuple<torch::Tensor, torch::Tensor> ROIHeadsImpl::_sample_proposals(const torch::Tensor &matched_idxs,
	const torch::Tensor &matched_labels, torch::Tensor gt_classes) {
	// Get the corresponding GT for each proposal
	if (gt_classes.numel() > 0) {
		gt_classes = gt_classes[matched_idxs];
		// Label unmatched proposals (0 label from matcher) as background (label=num_classes)
		gt_classes.index_put_({ matched_labels == 0 }, m_num_classes);
		// Label ignore proposals (-1 label)
		gt_classes.index_put_({ matched_labels == -1 }, -1);
	}
	else {
		gt_classes = torch::zeros_like(matched_idxs) + m_num_classes;
	}

	torch::Tensor sampled_fg_idxs, sampled_bg_idxs;
	tie(sampled_fg_idxs, sampled_bg_idxs) = subsample_labels(
		gt_classes, m_batch_size_per_image, m_positive_sample_fraction, m_num_classes
	);

	auto sampled_idxs = torch::cat({ sampled_fg_idxs, sampled_bg_idxs }, 0);
	return { sampled_idxs, gt_classes.index(sampled_idxs) };
}

InstancesList ROIHeadsImpl::label_and_sample_proposals(InstancesList &proposals, const InstancesList &targets) {
	torch::NoGradGuard guard;

	auto gt_boxes = targets.getTensorVec("gt_boxes");
	// Augment proposals with ground-truth boxes.
	// In the case of learned proposals (e.g., RPN), when training starts
	// the proposals will be low quality due to random initialization.
	// It's possible that none of these initial
	// proposals have high enough overlap with the gt objects to be used
	// as positive examples for the second stage components (box head,
	// cls head, mask head). Adding the gt boxes to the set of proposals
	// ensures that the second stage components will have some positive
	// examples from the start of training. For RPN, this augmentation improves
	// convergence and empirically improves box AP on COCO by about 0.5
	// points (under one tested configuration).
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
		auto match_quality_matrix = Boxes::pairwise_iou(
			targets_per_image->getTensor("gt_boxes"), proposals_per_image->getTensor("proposal_boxes")
		);
		Tensor matched_idxs, matched_labels;
		tie(matched_idxs, matched_labels) = m_proposal_matcher(match_quality_matrix);
		Tensor sampled_idxs, gt_classes;
		tie(sampled_idxs, gt_classes) = _sample_proposals(
			matched_idxs, matched_labels, targets_per_image->getTensor("gt_classes")
		);

		// Set target attributes of the sampled proposals:
		proposals_per_image = (*proposals_per_image)[sampled_idxs];
		proposals_per_image->set("gt_classes", gt_classes);

		// We index all the attributes of targets that start with "gt_"
		// and have not been added to proposals yet (="gt_classes").
		if (has_gt) {
			auto sampled_targets = matched_idxs.index(sampled_idxs);
			// NOTE: here the indexing waste some compute, because heads
			// like masks, keypoints, etc, will filter the proposals again,
			// (by foreground/background, or number of keypoints in the image, etc)
			// so we essentially index the data twice.
			auto &fields = targets_per_image->get_fields();
			for (auto iter : fields) {
				auto trg_name = iter.first;
				auto trg_value = iter.second;
				if (trg_name.find("gt_") == 0 && !proposals_per_image->has(trg_name)) {
					proposals_per_image->set(trg_name, trg_value->index(sampled_targets));
				}
			}
		}
		else {
			auto gt_boxes = targets_per_image->getTensor("gt_boxes").new_zeros({ sampled_idxs.size(0), 4 });
			proposals_per_image->set("gt_boxes", gt_boxes);
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

void ROIHeadsImpl::add_ground_truth_to_proposals(InstancesList &proposals, const BoxesList &gt_boxes) {
	int count = proposals.size();
	assert(gt_boxes.size() == count);
	for (int i = 0; i < count; i++) {
		auto &gt_boxes_i = gt_boxes[i];
		auto proposals_i = proposals[i];
		proposals[i] = add_ground_truth_to_proposals_single_image(gt_boxes_i, proposals_i);
	}
}

InstancesPtr ROIHeadsImpl::add_ground_truth_to_proposals_single_image(const torch::Tensor &gt_boxes,
	InstancesPtr proposals) {
	auto device = proposals->getTensor("objectness_logits").device();
	// Concatenating gt_boxes with proposals requires them to have the same fields
	// Assign all ground-truth boxes an objectness logit corresponding to P(object) \approx 1.
	auto gt_logit_value = log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)));

	auto gt_logits = gt_logit_value * torch::ones(gt_boxes.size(0), device);
	auto gt_proposal = make_shared<Instances>(proposals->image_size());

	gt_proposal->set("proposal_boxes", gt_boxes);
	gt_proposal->set("objectness_logits", gt_logits);
	auto new_proposals = proposals->cat({ proposals, gt_proposal }, proposals->size() + gt_proposal->size());

	return dynamic_pointer_cast<Instances>(new_proposals);
}
