#include "Base.h"
#include "RRPN.h"

#include <Detectron2/Structures/NMS.h>
#include <Detectron2/Structures/RotatedBoxes.h>
#include "RPNOutputs.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RRPNImpl::RRPNImpl(CfgNode &cfg, const ShapeSpec::Map &input_shapes) : RPNImpl(cfg, input_shapes) {
	m_box2box_transform = Box2BoxTransform::Create(cfg["MODEL.RPN.BBOX_REG_WEIGHTS"]);
	assert(m_boundary_threshold < 0); // boundary_threshold is a legacy option not implemented for RRPN
}

std::tuple<TensorVec, BoxesList> RRPNImpl::label_and_sample_anchors(const BoxesList &anchors_,
	const InstancesList &gt_instances) {
	torch::NoGradGuard guard;

	RotatedBoxes anchors = Boxes::cat(anchors_);

	auto gt_boxes = gt_instances.getTensorVec("gt_boxes");

	TensorVec gt_labels;
	BoxesList matched_gt_boxes;
	int count = gt_boxes.size();
	for (int i = 0; i < count; i++) {
		auto gt_boxes_i = gt_boxes[i]; // ground-truth boxes for i-th image

		Tensor match_quality_matrix;
		retry_if_cuda_oom([&]() {
			match_quality_matrix = RotatedBoxes::pairwise_iou_rotated(gt_boxes_i, anchors);
		});
		Tensor matched_idxs, gt_labels_i;
		retry_if_cuda_oom([&]() {
			tie(matched_idxs, gt_labels_i) = m_anchor_matcher(match_quality_matrix);
		});
		// Matching is memory-expensive and may result in CPU tensors. But the result is small
		gt_labels_i = gt_labels_i.to(gt_boxes_i.device());

        // A vector of labels (-1, 0, 1) for each anchor
		gt_labels_i = _subsample_labels(gt_labels_i);

		Tensor matched_gt_boxes_i;
		if (gt_boxes_i.numel() == 0) {
			// These values won't be used anyway since the anchor is labeled as background
			matched_gt_boxes_i = torch::zeros_like(anchors.tensor());
		}
		else {
			// TODO wasted indexing computation for ignored boxes
			matched_gt_boxes_i = gt_boxes_i.index(matched_idxs);
		}

		gt_labels.push_back(gt_labels_i);  // N,AHW
		matched_gt_boxes.push_back(matched_gt_boxes_i);
	}
	return { gt_labels, matched_gt_boxes };
}

InstancesList RRPNImpl::find_top_proposals(const TensorVec &proposals,
	const TensorVec &pred_objectness_logits, const ImageList &images, float nms_thresh,
	int pre_nms_topk, int post_nms_topk, float min_box_side_len, bool training) {
	auto image_sizes = images.image_sizes();  // in (h, w) order
	auto num_images = image_sizes.size();
	auto device = proposals[0].device();

	// 1. Select top-k anchor for every level and every image
	int count = proposals.size();
	TensorVec topk_scores; topk_scores.reserve(count); // #lvl Tensor, each of shape N x topk
	TensorVec topk_proposals; topk_proposals.reserve(count);
	TensorVec level_ids; level_ids.reserve(count); // #lvl Tensor, each of shape (topk,)
	auto batch_idx = torch::arange((int)num_images, device);
	for (int level_id = 0; level_id < count; level_id++) {
		auto proposals_i = proposals[level_id];
		auto logits_i = pred_objectness_logits[level_id];
		int Hi_Wi_A = logits_i.size(1);
		auto num_proposals_i = min(pre_nms_topk, Hi_Wi_A);

		// sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
		// topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
		Tensor idx;
		tie(logits_i, idx) = logits_i.sort(1, true);
		auto topk_scores_i = logits_i.index({ batch_idx, Slice(None, num_proposals_i) });
		auto topk_idx = idx.index({ batch_idx, Slice(None, num_proposals_i) });

		// each is N x topk
		auto topk_proposals_i = proposals_i.index({
			batch_idx.index({ Colon, None }), topk_idx });  // N x topk x 5

		topk_proposals.push_back(topk_proposals_i);
		topk_scores.push_back(topk_scores_i);
		level_ids.push_back(torch::full({ num_proposals_i },
			level_id, dtype(torch::kInt64).device(device)));
	}

	// 2. Concat all levels together
	auto t_topk_scores = torch::cat(topk_scores, 1);
	auto t_topk_proposals = torch::cat(topk_proposals, 1);
	auto t_level_ids = torch::cat(level_ids);

	// 3. For each image, run a per-level NMS, and choose topk results.
	InstancesList results;
	results.reserve(image_sizes.size());
	for (int n = 0; n < image_sizes.size(); n++) {
		auto &image_size = image_sizes[n];
		auto boxes = RotatedBoxes(t_topk_proposals[n]);
		auto scores_per_img = t_topk_scores[n];
		auto lvl = t_level_ids;

		auto valid_mask = torch::isfinite(boxes.tensor()).all(1).bitwise_and(torch::isfinite(scores_per_img));
		if (!valid_mask.all().item<bool>()) {
			boxes = boxes.tensor().index(valid_mask);
			scores_per_img = scores_per_img.index(valid_mask);
			lvl = lvl.index(valid_mask);
		}
		boxes.clip(image_size);

		// filter empty boxes
		auto keep = boxes.nonempty(min_box_side_len);
		lvl = t_level_ids;
		if (keep.sum().item<int64_t>() != boxes.len()) {
			boxes = boxes.tensor().index(keep);
			scores_per_img = scores_per_img.index(keep);
			lvl = lvl.index(keep);
		}

		keep = batched_nms_rotated(boxes.tensor(), scores_per_img, lvl, nms_thresh);
		// In Detectron1, there was different behavior during training vs. testing.
		// (https://github.com/facebookresearch/Detectron/issues/459)
		// During training, topk is over the proposals from *all* images in the training batch.
		// During testing, it is over the proposals for each image separately.
		// As a result, the training behavior becomes batch-dependent,
		// and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
		// This bug is addressed in Detectron2 to make the behavior independent of batch size.
		keep = keep.index({ Slice(None, post_nms_topk) });

		auto res = make_shared<Instances>(image_size);
		res->set("proposal_boxes", boxes.tensor().index(keep));
		res->set("objectness_logits", scores_per_img.index(keep));
		results.push_back(res);
	}
	return results;
}
