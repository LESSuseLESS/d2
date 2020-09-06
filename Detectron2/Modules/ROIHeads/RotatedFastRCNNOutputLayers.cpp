#include "Base.h"
#include "RotatedFastRCNNOutputLayers.h"

#include <Detectron2/Structures/NMS.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<InstancesList, TensorVec> Detectron2::fast_rcnn_inference_rotated(const TensorVec &boxes,
	const TensorVec &scores, const std::vector<ImageSize> &image_shapes, float score_thresh,
	float nms_thresh, int topk_per_image) {
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
		tie(instances, boxes) = fast_rcnn_inference_single_image_rotated(
			boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
		);
		instances_list.push_back(instances);
		boxes_list.push_back(boxes);
	}
	return { instances_list, boxes_list };
}

std::tuple<InstancesPtr, torch::Tensor> Detectron2::fast_rcnn_inference_single_image_rotated(
	const torch::Tensor &boxes_, const torch::Tensor &scores_, const ImageSize &image_shape,
	float score_thresh, float nms_thresh, int topk_per_image) {
	torch::Tensor t_boxes = boxes_;
	torch::Tensor scores = scores_;
	auto valid_mask = torch::isfinite(t_boxes).all(1).bitwise_and(torch::isfinite(scores).all(1));
	if (!valid_mask.all().item<bool>()) {
		t_boxes = t_boxes.index(valid_mask);
		scores = scores.index(valid_mask);
	}

	int B = 5;  // box dimension
	scores = scores.index({ Colon, Slice(None, -1) });
	auto num_bbox_reg_classes = t_boxes.size(1) / B;
	// Convert to Boxes to use the `clip` function ...
	auto boxes = Boxes::boxes(t_boxes.reshape({ -1, B }));
	boxes->clip(image_shape);
	t_boxes = boxes->tensor().view({ -1, num_bbox_reg_classes, B });  // R x C x B

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

	// Apply per-class Rotated NMS
	auto keep = batched_nms_rotated(t_boxes, scores, filter_inds.index({ Colon, 1 }), nms_thresh);
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

RotatedFastRCNNOutputLayersImpl::RotatedFastRCNNOutputLayersImpl(CfgNode &cfg, const ShapeSpec &input_shape) :
	FastRCNNOutputLayersImpl(cfg, input_shape) {
	m_box2box_transform = Box2BoxTransform::Create(cfg["MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS"]);
}

std::tuple<InstancesList, TensorVec> RotatedFastRCNNOutputLayersImpl::inference(const TensorVec &predictions,
	const InstancesList &proposals) {
    auto boxes = predict_boxes(predictions, proposals);
    auto scores = predict_probs(predictions, proposals);
	auto image_shapes = proposals.getImageSizes();

    return fast_rcnn_inference_rotated(boxes, scores, image_shapes,
        m_test_score_thresh, m_test_nms_thresh, m_test_topk_per_image);
}