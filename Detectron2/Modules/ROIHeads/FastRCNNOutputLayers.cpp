#include "Base.h"
#include "FastRCNNOutputLayers.h"
#include "FastRCNNOutputs.h"

#include <Detectron2/Structures/NMS.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<InstancesList, TensorVec> Detectron2::fast_rcnn_inference(const TensorVec &boxes, const TensorVec &scores,
	const std::vector<ImageSize> &image_shapes, float score_thresh, float nms_thresh, int topk_per_image) {
	InstancesList instances_list;
	TensorVec boxes_list;

	int count = scores.size();
	assert(boxes.size() == count);
	assert(image_shapes.size() == count);
	instances_list.reserve(count);
	boxes_list.reserve(count);
	for (int i = 0; i < count; i++) {
		auto scores_per_image = scores[i];
		auto boxes_per_image = boxes[i];
		auto image_shape = image_shapes[i];

		InstancesPtr instances;
		torch::Tensor boxes;
		tie(instances, boxes) = fast_rcnn_inference_single_image(
			boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
		);
		instances_list.push_back(instances);
		boxes_list.push_back(boxes);
	}
	return { instances_list, boxes_list };
}

std::tuple<InstancesPtr, torch::Tensor> Detectron2::fast_rcnn_inference_single_image(
	const torch::Tensor &boxes_, const torch::Tensor &scores_, const ImageSize &image_shape,
	float score_thresh, float nms_thresh, int topk_per_image) {
	torch::Tensor t_boxes = boxes_;
	torch::Tensor scores = scores_;
	auto valid_mask = torch::isfinite(t_boxes).all(1).bitwise_and(torch::isfinite(scores).all(1));
	if (!valid_mask.all().item<bool>()) {
		t_boxes = t_boxes.index(valid_mask);
		scores = scores.index(valid_mask);
	}

	scores = scores.index({ Colon, Slice(None, -1) });
	auto num_bbox_reg_classes = t_boxes.size(1) / 4;
	// Convert to Boxes to use the `clip` function ...
	auto boxes = Boxes::boxes(t_boxes.reshape({ -1, 4 }));
	boxes->clip(image_shape);
	t_boxes = boxes->tensor().view({ -1, num_bbox_reg_classes, 4 });  // R x C x 4

	// Filter results based on detection scores
	auto filter_mask = scores > score_thresh;  // R x K
	// R' x 2. First column contains indices of the R predictions;
	// Second column contains indices of classes.
	auto filter_inds = filter_mask.nonzero();
	if (num_bbox_reg_classes == 1) {
		t_boxes = t_boxes.index({ filter_inds.index({ Colon, 0 }), 0 });
	}
	else {
		t_boxes = t_boxes.index(filter_mask);
	}
	scores = scores.index(filter_mask);

	// Apply per-class NMS
	auto keep = batched_nms(t_boxes, scores, filter_inds.index({ Colon, 1 }), nms_thresh);
	if (topk_per_image >= 0) {
		keep = keep.index({ Slice(None, topk_per_image) });
	}
	t_boxes = t_boxes.index(keep);
	scores = scores.index(keep);
	filter_inds = filter_inds.index(keep);

	auto result = make_shared<Instances>(image_shape);
	result->set("pred_boxes", t_boxes);
	result->set("scores", scores);
	result->set("pred_classes", filter_inds.index({ Colon, 1 }));
	return { result, filter_inds.index({ Colon, 0 }) };
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FastRCNNOutputLayersImpl::FastRCNNOutputLayersImpl(CfgNode &cfg, const ShapeSpec &input_shape,
	const std::shared_ptr<Box2BoxTransform> &box2box_transform) :
	m_smooth_l1_beta(cfg["MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA"].as<float>()),
	m_test_score_thresh(cfg["MODEL.ROI_HEADS.SCORE_THRESH_TEST"].as<float>()),
	m_test_nms_thresh(cfg["MODEL.ROI_HEADS.NMS_THRESH_TEST"].as<float>()),
	m_test_topk_per_image(cfg["TEST.DETECTIONS_PER_IMAGE"].as<int>())
{
	m_box2box_transform = box2box_transform;
	if (!m_box2box_transform) {
		m_box2box_transform = Box2BoxTransform::Create(cfg["MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS"]);
	}

	auto num_classes = cfg["MODEL.ROI_HEADS.NUM_CLASSES"].as<int>();
	auto cls_agnostic = cfg["MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG"].as<bool>();

	// The prediction layer for num_classes foreground classes and one background class (hence + 1)
	m_cls_score = nn::Linear(input_shape.prod(), num_classes  + 1);
	register_module("cls_score", m_cls_score);
	m_bbox_pred = nn::Linear(input_shape.prod(), (cls_agnostic ? 1 : num_classes) * m_box2box_transform->box_dim());
	register_module("bbox_pred", m_bbox_pred);
}

void FastRCNNOutputLayersImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	importer.Import(prefix + ".cls_score", m_cls_score, ModelImporter::kNormalFill2);
	importer.Import(prefix + ".bbox_pred", m_bbox_pred, ModelImporter::kNormalFill3);
}

TensorVec FastRCNNOutputLayersImpl::forward(torch::Tensor x) {
	if (x.dim() > 2) {
		x = torch::flatten(x, 1);
	}
	auto scores = m_cls_score(x);
	auto proposal_deltas = m_bbox_pred(x);
	return { scores, proposal_deltas };
}

TensorMap FastRCNNOutputLayersImpl::losses(const TensorVec &predictions, const InstancesList &proposals) {
	auto scores = predictions[0];
	auto proposal_deltas = predictions[1];
	return FastRCNNOutputs(m_box2box_transform, scores, proposal_deltas, proposals, m_smooth_l1_beta).losses();
}

std::tuple<InstancesList, TensorVec> FastRCNNOutputLayersImpl::inference(const TensorVec &predictions,
	const InstancesList &proposals) {
	auto boxes = predict_boxes(predictions, proposals);
	auto scores = predict_probs(predictions, proposals);
	auto image_shapes = proposals.getImageSizes();
	return fast_rcnn_inference(boxes, scores, image_shapes,
		m_test_score_thresh, m_test_nms_thresh, m_test_topk_per_image);
}

TensorVec FastRCNNOutputLayersImpl::predict_boxes_for_gt_classes(const TensorVec &predictions,
	const InstancesList &proposals) {
	if (proposals.empty()) {
		return {};
	}

	auto scores = predictions[0];
	auto proposal_deltas = predictions[1];
	auto proposal_boxes_list = proposals.getTensorVec("proposal_boxes");
	auto proposal_boxes = torch::cat(proposal_boxes_list);
	auto N = proposal_boxes.size(0);
	auto B = proposal_boxes.size(1);
	auto predict_boxes = m_box2box_transform->apply_deltas_broadcast(proposal_deltas, proposal_boxes); // Nx(KxB)

	auto K = predict_boxes.size(1) / B;
	if (K > 1) {
		auto gt_classes_list = proposals.getTensorVec("gt_classes");
		auto gt_classes = torch::cat(gt_classes_list, 0);
		// Some proposals are ignored or have a background class. Their gt_classes
		// cannot be used as index.
		gt_classes = gt_classes.clamp_(0, K - 1);

		predict_boxes = predict_boxes.view({ N, K, B }).index(
			{ torch::arange(N, dtype(torch::kLong).device(predict_boxes.device())), gt_classes }
		);
	}
	auto num_prop_per_image = proposals.getLenVec();
	return predict_boxes.split_with_sizes(num_prop_per_image);
}

TensorVec FastRCNNOutputLayersImpl::predict_boxes(const TensorVec &predictions, const InstancesList &proposals) {
	if (proposals.empty()) {
		return {};
	}

	auto proposal_deltas = predictions[1];
	auto num_prop_per_image = proposals.getLenVec();
	auto proposal_boxes_list = proposals.getTensorVec("proposal_boxes");
	auto proposal_boxes = torch::cat(proposal_boxes_list);
	auto predict_boxes = m_box2box_transform->apply_deltas_broadcast(proposal_deltas, proposal_boxes); // Nx(KxB)
	return predict_boxes.split_with_sizes(num_prop_per_image);
}

TensorVec FastRCNNOutputLayersImpl::predict_probs(const TensorVec &predictions, const InstancesList &proposals) {
	auto scores = predictions[0];
	auto num_inst_per_image = proposals.getLenVec();
	auto probs = nn::functional::softmax(scores, -1);
	return probs.split_with_sizes(num_inst_per_image, 0);
}
