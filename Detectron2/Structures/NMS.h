#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from layers/nms.py
	// converted from https://github.com/pytorch/vision torchvision/ops/boxes.py

	namespace torchvision {

		/**
			Performs non-maximum suppression (NMS) on the boxes according
			to their intersection-over-union (IoU).

			NMS iteratively removes lower scoring boxes which have an
			IoU greater than iou_threshold with another (higher scoring)
			box.

			If multiple boxes have the exact same score and satisfy the IoU
			criterion with respect to a reference box, the selected box is
			not guaranteed to be the same between CPU and GPU. This is similar
			to the behavior of argsort in PyTorch when repeated values are present.

			Parameters
			----------
			boxes : Tensor[N, 4])
				boxes to perform NMS on. They
				are expected to be in (x1, y1, x2, y2) format
			scores : Tensor[N]
				scores for each one of the boxes
			iou_threshold : float
				discards all overlapping
				boxes with IoU > iou_threshold

			Returns
			-------
			keep : Tensor
				int64 tensor with the indices
				of the elements that have been kept
				by NMS, sorted in decreasing order of scores
		*/
		torch::Tensor nms(const torch::Tensor &boxes, const torch::Tensor &scores, float iou_threshold);

		/**
			Performs non-maximum suppression in a batched fashion.

			Each index value correspond to a category, and NMS
			will not be applied between elements of different categories.

			Parameters
			----------
			boxes : Tensor[N, 4]
				boxes where NMS will be performed. They
				are expected to be in (x1, y1, x2, y2) format
			scores : Tensor[N]
				scores for each one of the boxes
			idxs : Tensor[N]
				indices of the categories for each one of the boxes.
			iou_threshold : float
				discards all overlapping boxes
				with IoU > iou_threshold

			Returns
			-------
			keep : Tensor
				int64 tensor with the indices of
				the elements that have been kept by NMS, sorted
				in decreasing order of scores
		*/
		torch::Tensor batched_nms(const torch::Tensor &boxes, const torch::Tensor &scores, const torch::Tensor &idxs,
			float iou_threshold);

	} // namespace torchvision

	// Same as torchvision.ops.boxes.batched_nms, but safer.
	torch::Tensor batched_nms(const torch::Tensor &boxes, const torch::Tensor &scores, const torch::Tensor &idxs,
		float iou_threshold);

	// Note: this function (nms_rotated) might be moved into
	// torchvision/ops/boxes.py in the future
    /**
		Performs non-maximum suppression (NMS) on the rotated boxes according
		to their intersection-over-union (IoU).

		Rotated NMS iteratively removes lower scoring rotated boxes which have an
		IoU greater than iou_threshold with another (higher scoring) rotated box.

		Note that RotatedBox (5, 3, 4, 2, -90) covers exactly the same region as
		RotatedBox (5, 3, 4, 2, 90) does, and their IoU will be 1. However, they
		can be representing completely different objects in certain tasks, e.g., OCR.

		As for the question of whether rotated-NMS should treat them as faraway boxes
		even though their IOU is 1, it depends on the application and/or ground truth annotation.

		As an extreme example, consider a single character v and the square box around it.

		If the angle is 0 degree, the object (text) would be read as 'v';

		If the angle is 90 degrees, the object (text) would become '>';

		If the angle is 180 degrees, the object (text) would become '^';

		If the angle is 270/-90 degrees, the object (text) would become '<'

		All of these cases have IoU of 1 to each other, and rotated NMS that only
		uses IoU as criterion would only keep one of them with the highest score -
		which, practically, still makes sense in most cases because typically
		only one of theses orientations is the correct one. Also, it does not matter
		as much if the box is only used to classify the object (instead of transcribing
		them with a sequential OCR recognition model) later.

		On the other hand, when we use IoU to filter proposals that are close to the
		ground truth during training, we should definitely take the angle into account if
		we know the ground truth is labeled with the strictly correct orientation (as in,
		upside-down words are annotated with -180 degrees even though they can be covered
		with a 0/90/-90 degree box, etc.)

		The way the original dataset is annotated also matters. For example, if the dataset
		is a 4-point polygon dataset that does not enforce ordering of vertices/orientation,
		we can estimate a minimum rotated bounding box to this polygon, but there's no way
		we can tell the correct angle with 100% confidence (as shown above, there could be 4 different
		rotated boxes, with angles differed by 90 degrees to each other, covering the exactly
		same region). In that case we have to just use IoU to determine the box
		proximity (as many detection benchmarks (even for text) do) unless there're other
		assumptions we can make (like width is always larger than height, or the object is not
		rotated by more than 90 degrees CCW/CW, etc.)

		In summary, not considering angles in rotated NMS seems to be a good option for now,
		but we should be aware of its implications.

		Args:
			boxes (Tensor[N, 5]): Rotated boxes to perform NMS on. They are expected to be in
			   (x_center, y_center, width, height, angle_degrees) format.
			scores (Tensor[N]): Scores for each one of the rotated boxes
			iou_threshold (float): Discards all overlapping rotated boxes with IoU < iou_threshold

		Returns:
			keep (Tensor): int64 tensor with the indices of the elements that have been kept
			by Rotated NMS, sorted in decreasing order of scores
	*/
	torch::Tensor nms_rotated(const torch::Tensor &boxes, const torch::Tensor &scores, float iou_threshold);

	// Note: this function (batched_nms_rotated) might be moved into
	// torchvision/ops/boxes.py in the future
    /**
		Performs non-maximum suppression in a batched fashion.

		Each index value correspond to a category, and NMS
		will not be applied between elements of different categories.

		Args:
			boxes (Tensor[N, 5]):
			   boxes where NMS will be performed. They
			   are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
			scores (Tensor[N]):
			   scores for each one of the boxes
			idxs (Tensor[N]):
			   indices of the categories for each one of the boxes.
			iou_threshold (float):
			   discards all overlapping boxes
			   with IoU < iou_threshold

		Returns:
			Tensor:
				int64 tensor with the indices of the elements that have been kept
				by NMS, sorted in decreasing order of scores
	*/
	torch::Tensor batched_nms_rotated(const torch::Tensor &boxes, const torch::Tensor &scores,
		const torch::Tensor &idxs, float iou_threshold);
}