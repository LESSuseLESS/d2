#include "Base.h"
#include "RPN.h"

#include <Detectron2/Structures/NMS.h>
#include <Detectron2/Structures/Sampling.h>
#include "RPNOutputs.h"
#include "RRPN.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RPN Detectron2::build_proposal_generator(CfgNode &cfg, const ShapeSpec::Map &input_shapes) {
	auto name = cfg["MODEL.PROPOSAL_GENERATOR.NAME"].as<string>();
	if (name == "PrecomputedProposals") {
		return nullptr;
	}
	if (name == "RPN") {
		return shared_ptr<RPNImpl>(new RPNImpl(cfg, input_shapes));
	}
	if (name == "RRPN") {
		return shared_ptr<RPNImpl>(new RRPNImpl(cfg, input_shapes));
	}
	assert(false);
	return nullptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor RPNImpl::_subsample_labels(torch::Tensor label) {
	Tensor pos_idx, neg_idx;
	tie(pos_idx, neg_idx) = subsample_labels(label, m_batch_size_per_image, m_positive_fraction, 0);
	// Fill with the ignore label (-1), then set positive and negative labels
	label.fill_(-1);
	label.scatter_(0, pos_idx, 1);
	label.scatter_(0, neg_idx, 0);
	return label;
}

std::tuple<TensorVec, BoxesList> RPNImpl::label_and_sample_anchors(const BoxesList &anchors_,
	const InstancesList &gt_instances) {
	torch::NoGradGuard guard;

	Boxes anchors = Boxes::cat(anchors_);

	auto gt_boxes = gt_instances.getTensorVec("gt_boxes");
	auto image_sizes = gt_instances.getImageSizes();

	TensorVec gt_labels;
	BoxesList matched_gt_boxes;
	int count = image_sizes.size();
	assert(gt_boxes.size() == count);
	for (int i = 0; i < count; i++) {
		// (h, w) for the i-th image
		ImageSize &image_size_i = image_sizes[i];
		auto gt_boxes_i = gt_boxes[i]; // ground-truth boxes for i-th image

		Tensor match_quality_matrix;
		retry_if_cuda_oom([&]() {
			match_quality_matrix = Boxes::pairwise_iou(gt_boxes_i, anchors);
		});
		Tensor matched_idxs, gt_labels_i;
		retry_if_cuda_oom([&]() {
			tie(matched_idxs, gt_labels_i) = m_anchor_matcher(match_quality_matrix);
		});
		// Matching is memory-expensive and may result in CPU tensors. But the result is small
		gt_labels_i = gt_labels_i.to(gt_boxes_i.device());

		if (m_boundary_threshold >= 0) {
			// Discard anchors that go out of the boundaries of the image
			// NOTE: This is legacy functionality that is turned off by default in Detectron2
			auto anchors_inside_image = anchors.inside_box(image_size_i, m_boundary_threshold);
			gt_labels_i.index_put_({ anchors_inside_image.neg() }, -1);
		}

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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RPNImpl::RPNImpl(CfgNode &cfg, const ShapeSpec::Map &input_shapes) :
	m_in_features(cfg["MODEL.RPN.IN_FEATURES"].as<vector<string>>()),
	m_anchor_matcher(
		cfg["MODEL.RPN.IOU_THRESHOLDS"].as<vector<float>>(),
		cfg["MODEL.RPN.IOU_LABELS"].as<vector<int>>(),
		true // allow_low_quality_matches
	),
	m_min_box_side_len(cfg["MODEL.PROPOSAL_GENERATOR.MIN_SIZE"].as<int>()),
	m_nms_thresh(cfg["MODEL.RPN.NMS_THRESH"].as<float>()),
	m_batch_size_per_image(cfg["MODEL.RPN.BATCH_SIZE_PER_IMAGE"].as<int>()),
	m_positive_fraction(cfg["MODEL.RPN.POSITIVE_FRACTION"].as<float>()),
	m_smooth_l1_beta(cfg["MODEL.RPN.SMOOTH_L1_BETA"].as<float>()),
	m_loss_weight(cfg["MODEL.RPN.LOSS_WEIGHT"].as<float>()),
	m_pre_nms_topk{ // Map from m_training state to train / test settings
		/* false : */ cfg["MODEL.RPN.PRE_NMS_TOPK_TEST"].as<int>(),
		/* true: */ cfg["MODEL.RPN.PRE_NMS_TOPK_TRAIN"].as<int>()
	},
	m_post_nms_topk{
		/* false : */ cfg["MODEL.RPN.POST_NMS_TOPK_TEST"].as<int>(),
		/* true: */ cfg["MODEL.RPN.POST_NMS_TOPK_TRAIN"].as<int>()
	},
	m_boundary_threshold(cfg["MODEL.RPN.BOUNDARY_THRESH"].as<int>())
{
	m_box2box_transform = Box2BoxTransform::Create(cfg["MODEL.RPN.BBOX_REG_WEIGHTS"]);

	auto filtered = ShapeSpec::filter(input_shapes, m_in_features);
	m_anchor_generator = build_anchor_generator(cfg, filtered);
	register_module("anchor_generator", m_anchor_generator);
	m_rpn_head = build_rpn_head(cfg, filtered);
	register_module("rpn_head", m_rpn_head);
}

void RPNImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	m_anchor_generator->initialize(importer, prefix + ".anchor_generator");
	m_rpn_head->initialize(importer, prefix + ".rpn_head");
}

std::tuple<InstancesList, TensorMap> RPNImpl::forward(const ImageList &images,
	const TensorMap &features_, const InstancesList &gt_instances) {
	TensorVec features;
	for (auto f : m_in_features) {
		auto iter = features_.find(f);
		assert(iter != features_.end());
		features.push_back(iter->second);
	}
	auto res = m_rpn_head->forward(features);
	auto pred_objectness_logits = res[0];
	auto pred_anchor_deltas = res[1];

	auto anchors = m_anchor_generator->forward(features);

	TensorVec gt_labels;
	BoxesList gt_boxes;
	if (is_training()) {
		tie(gt_labels, gt_boxes) = label_and_sample_anchors(anchors, gt_instances);
	}

	RPNOutputs outputs(m_box2box_transform, m_batch_size_per_image, images,
		pred_objectness_logits, pred_anchor_deltas,
		std::move(anchors), std::move(gt_labels), std::move(gt_boxes), m_smooth_l1_beta);

	TensorMap losses;
	if (is_training()) {
		auto items = outputs.losses();
		for (auto iter : items) {
			losses[iter.first] = iter.second * m_loss_weight;
		}
	}

	InstancesList proposals;
	{
		torch::NoGradGuard guard;

		// Find the top proposals by applying NMS and removing boxes that
		// are too small. The proposals are treated as fixed for approximate
		// joint training with roi heads. This approach ignores the derivative
		// w.r.t. the proposal boxes’ coordinates that are also network
		// responses, so is approximate.
		proposals = find_top_proposals(
			outputs.predict_proposals(),
			outputs.predict_objectness_logits(),
			images,
			m_nms_thresh,
			m_pre_nms_topk[is_training() ? 1 : 0],
			m_post_nms_topk[is_training() ? 1 : 0],
			m_min_box_side_len,
			is_training());
	}
	return { proposals, losses };
}

InstancesList RPNImpl::find_top_proposals(const TensorVec &proposals,
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
		auto topk_proposals_i = proposals_i.index({ batch_idx.index({ Colon, None }), topk_idx });  // N x topk x 4

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
		auto boxes = Boxes(t_topk_proposals[n]);
		auto scores_per_img = t_topk_scores[n];
		auto lvl = t_level_ids;

		auto valid_mask = torch::isfinite(boxes.tensor()).all(1).bitwise_and(torch::isfinite(scores_per_img));
		if (!valid_mask.all().item<bool>()) {
			assert(!training); // FloatingPointError: Predicted boxes or scores contain Inf/NaN. Training has diverged
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

		keep = batched_nms(boxes.tensor(), scores_per_img, lvl, nms_thresh);
		// In Detectron1, there was different behavior during training vs. testing.
		// (https://github.com/facebookresearch/Detectron/issues/459)
		// During training, topk is over the proposals from *all* images in the training batch.
		// During testing, it is over the proposals for each image separately.
		// As a result, the training behavior becomes batch-dependent,
		// and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
		// This bug is addressed in Detectron2 to make the behavior independent of batch size.
		keep = keep.index({ Slice(None, post_nms_topk) });  // keep is already sorted

		auto res = make_shared<Instances>(image_size);
		res->set("proposal_boxes", boxes.tensor().index(keep));
		res->set("objectness_logits", scores_per_img.index(keep));
		results.push_back(res);
	}
	return results;
}
