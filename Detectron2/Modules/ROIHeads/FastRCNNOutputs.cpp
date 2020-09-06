#include "Base.h"
#include "FastRCNNOutputs.h"

#include <Detectron2/fvcore/fvcore.h>
#include <Detectron2/Utils/EventStorage.h>
#include "FastRCNNOutputLayers.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FastRCNNOutputs::FastRCNNOutputs(const std::shared_ptr<Box2BoxTransform> &box2box_transform,
	const torch::Tensor &pred_class_logits, const torch::Tensor &pred_proposal_deltas, const InstancesList &proposals,
	float smooth_l1_beta) :
	m_box2box_transform(box2box_transform),
	m_pred_class_logits(pred_class_logits),
	m_pred_proposal_deltas(pred_proposal_deltas),
	m_smooth_l1_beta(smooth_l1_beta)
{
	m_num_preds_per_image = proposals.getLenVec();
	m_image_shapes = proposals.getImageSizes();

	if (!proposals.empty()) {
		// cat(..., dim=0) concatenates over all images in the batch
		auto boxes = proposals.getTensorVec("proposal_boxes");
		m_proposals = torch::cat(boxes);
		assert(!m_proposals.requires_grad()); // Proposals should not require gradients!

		// The following fields should exist only when training.
		if (proposals[0]->has("gt_boxes")) {
			auto boxes = proposals.getTensorVec("gt_boxes");
			m_gt_boxes = torch::cat(boxes);
			assert(proposals[0]->has("gt_classes"));
			auto classes = proposals.getTensorVec("gt_classes");
			m_gt_classes = torch::cat(classes);
		}
	}
	else {
		m_proposals = torch::zeros({ 0, 4 }, m_pred_proposal_deltas.device());
	}
	m_no_instances = proposals.empty(); // no instances found
}

void FastRCNNOutputs::_log_accuracy() {
	auto num_instances = m_gt_classes.numel();
	auto pred_classes = m_pred_class_logits.argmax(1);
	auto bg_class_ind = m_pred_class_logits.size(1) - 1;

	auto fg_inds = (m_gt_classes >= 0).bitwise_and(m_gt_classes < bg_class_ind);
	auto num_fg = fg_inds.nonzero().numel();
	auto fg_gt_classes = m_gt_classes.index(fg_inds);
	auto fg_pred_classes = pred_classes.index(fg_inds);

	auto num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel();
	auto num_accurate = (pred_classes == m_gt_classes).nonzero().numel();
	auto fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel();

	auto &storage = get_event_storage();
	if (num_instances > 0) {
		storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances);
		if (num_fg > 0) {
			storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg);
			storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg);
		}
	}
}

torch::Tensor FastRCNNOutputs::softmax_cross_entropy_loss() {
	if (m_no_instances) {
		return 0.0 * m_pred_class_logits.sum();
	}
	else {
		_log_accuracy();
		return nn::functional::cross_entropy(m_pred_class_logits, m_gt_classes,
			nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
	}
}

torch::Tensor FastRCNNOutputs::smooth_l1_loss() {
	if (m_no_instances) {
		return 0.0 * m_pred_proposal_deltas.sum();
	}
	auto gt_proposal_deltas = m_box2box_transform->get_deltas(m_proposals, m_gt_boxes);
	auto box_dim = gt_proposal_deltas.size(1); // 4 or 5
	auto cls_agnostic_bbox_reg = (m_pred_proposal_deltas.size(1) == box_dim);
	auto device = m_pred_proposal_deltas.device();

	auto bg_class_ind = m_pred_class_logits.size(1) - 1;

    // Box delta loss is only computed between the prediction for the gt class k
    // (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
    // for non-gt classes and background.
    // Empty fg_inds produces a valid loss of zero as long as the size_average
    // arg to smooth_l1_loss is False (otherwise it uses torch::mean internally
    // and would produce a nan loss).
	auto fg_inds = torch::nonzero((m_gt_classes >= 0).bitwise_and(m_gt_classes < bg_class_ind)).index({ Colon, 0 });
	Tensor gt_class_cols;
	if (cls_agnostic_bbox_reg) {
		// pred_proposal_deltas only corresponds to foreground class for agnostic
		gt_class_cols = torch::arange(box_dim, device);
	}
	else {
		auto fg_gt_classes = m_gt_classes.index(fg_inds);
		// pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
		// where b is the dimension of box representation (4 or 5)
		// Note that compared to Detectron1,
		// we do not perform bounding box regression for background classes.
		gt_class_cols = box_dim * fg_gt_classes.index({ Colon, None }) +
			torch::arange(box_dim, device);
	}

	auto loss_box_reg = fvcore::smooth_l1_loss(
		m_pred_proposal_deltas.index({ fg_inds.index({ Colon, None }), gt_class_cols }),
		gt_proposal_deltas.index(fg_inds),
		m_smooth_l1_beta,
		torch::Reduction::Sum
	);
    // The loss is normalized using the total number of regions (R), not the number
    // of foreground regions even though the box regression loss is only defined on
    // foreground regions. Why? Because doing so gives equal training influence to
    // each foreground example. To see how, consider two different minibatches:
    //  (1) Contains a single foreground region
    //  (2) Contains 100 foreground regions
    // If we normalize by the number of foreground regions, the single example in
    // minibatch (1) will be given 100 times as much influence as each foreground
    // example in minibatch (2). Normalizing by the total number of regions, R,
    // means that the single example in minibatch (1) and each of the 100 examples
    // in minibatch (2) are given equal influence.
	loss_box_reg = loss_box_reg / m_gt_classes.numel();
	return loss_box_reg;
}

torch::Tensor FastRCNNOutputs::_predict_boxes() {
	return m_box2box_transform->apply_deltas_broadcast(m_pred_proposal_deltas, m_proposals);
}

TensorMap FastRCNNOutputs::losses() {
	return {
		{ "loss_cls", softmax_cross_entropy_loss() },
		{ "loss_box_reg", smooth_l1_loss() },
	};
}

TensorVec FastRCNNOutputs::predict_boxes() {
	return _predict_boxes().split_with_sizes(m_num_preds_per_image, 0);
}

TensorVec FastRCNNOutputs::predict_probs() {
	auto probs = nn::functional::softmax(m_pred_class_logits, -1);
	return probs.split_with_sizes(m_num_preds_per_image, 0);
}

std::tuple<InstancesList, TensorVec> FastRCNNOutputs::inference(float score_thresh, float nms_thresh,
	int topk_per_image) {
	auto boxes = predict_boxes();
	auto scores = predict_probs();
	return fast_rcnn_inference(boxes, scores, m_image_shapes, score_thresh, nms_thresh, topk_per_image);
}
