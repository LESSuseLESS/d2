#include "Base.h"
#include "NMS.h"

#include <Detectron2/detectron2/nms/nms.h>
#include <Detectron2/detectron2/nms_rotated/nms_rotated.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// namespace torchvision

torch::Tensor torchvision::nms(const torch::Tensor &boxes, const torch::Tensor &scores, float iou_threshold) {
	return detectron2::nms(boxes, scores, iou_threshold);
}
torch::Tensor torchvision::batched_nms(const torch::Tensor &boxes, const torch::Tensor &scores,
	const torch::Tensor &idxs, float iou_threshold) {
	if (boxes.numel() == 0) {
		return torch::empty({ 0 }, dtype(torch::kInt64).device(boxes.device()));
	}
	// strategy: in order to perform NMS independently per class.
	// we add an offset to all the boxes. The offset is dependent
	// only on the class idx, and is large enough so that boxes
	// from different classes do not overlap
	else {
		auto max_coordinate = boxes.max();
		auto offsets = idxs.to(boxes) * (max_coordinate + torch::tensor(1).to(boxes));
		auto boxes_for_nms = boxes + offsets.index({ Colon, None });
		auto keep = torchvision::nms(boxes_for_nms, scores, iou_threshold);
		return keep;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor Detectron2::batched_nms(const torch::Tensor &boxes, const torch::Tensor &scores,
	const torch::Tensor &idxs, float iou_threshold) {
	assert(boxes.size(-1) == 4);
	// TODO may need better strategy.
	// Investigate after having a fully-cuda NMS op.
	if (boxes.size(0) < 40000) {
		return torchvision::batched_nms(boxes, scores, idxs, iou_threshold);
	}

	auto result_mask = scores.new_zeros(scores.size(0), torch::kBool);
	auto unique_idxs = tolist(get<0>(torch::_unique2(idxs)).cpu());
	for (int i = 0; i < unique_idxs.size(0); i++) {
		auto id = unique_idxs[i];
		auto mask = (idxs == id).nonzero().view(-1);
		auto keep = torchvision::nms(boxes[mask], scores[mask], iou_threshold);
		result_mask[mask[keep]] = true;
	}
	auto keep = result_mask.nonzero().view(-1);
	keep = keep.index(scores.index(keep)).argsort(-1, true);
	return keep;
}

torch::Tensor Detectron2::nms_rotated(const torch::Tensor &boxes, const torch::Tensor &scores, float iou_threshold) {
	return detectron2::nms_rotated(boxes, scores, iou_threshold);
}

torch::Tensor Detectron2::batched_nms_rotated(const torch::Tensor &boxes, const torch::Tensor &scores,
	const torch::Tensor &idxs, float iou_threshold) {
	assert(boxes.size(-1) == 5);

	if (boxes.numel() == 0) {
		return torch::empty({ 0, -1 }, dtype(torch::kInt64).device(boxes.device()));
	}
	// Strategy: in order to perform NMS independently per class,
	// we add an offset to all the boxes. The offset is dependent
	// only on the class idx, and is large enough so that boxes
	// from different classes do not overlap

	// Note that batched_nms in torchvision/ops/boxes.py only uses max_coordinate,
	// which won't handle negative coordinates correctly.
	// Here by using min_coordinate we can make sure the negative coordinates are
	// correctly handled.
	auto max_coordinate = (
		torch::max(boxes.index({ Colon, 0 }), boxes.index({ Colon, 1 })) +
		torch::max(boxes.index({ Colon, 2 }), boxes.index({ Colon, 3 })) / 2
		).max();
	auto min_coordinate = (
		torch::min(boxes.index({ Colon, 0 }), boxes.index({ Colon, 1})) -
		torch::max(boxes.index({ Colon, 2 }), boxes.index({ Colon, 3 })) / 2
		).min();
	auto offsets = idxs.to(boxes) * (max_coordinate - min_coordinate + 1);
	auto boxes_for_nms = boxes.clone(); // avoid modifying the original values in boxes
	boxes_for_nms.index_put_({ Colon, Slice(None, 2) }, 
		boxes_for_nms.index({ Colon, Slice(None, 2) }) + offsets.index({ Colon, None }));
	auto keep = nms_rotated(boxes_for_nms, scores, iou_threshold);
	return keep;
}