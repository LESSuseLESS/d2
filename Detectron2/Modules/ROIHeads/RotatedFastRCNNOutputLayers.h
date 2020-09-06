#pragma once

#include "FastRCNNOutputLayers.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/roi_heads/rotated_fast_rcnn.py

	/**
		Shape shorthand in this module:

			N: number of images in the minibatch
			R: number of ROIs, combined over all images, in the minibatch
			Ri: number of ROIs in image i
			K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

		Naming convention:

			deltas: refers to the 5-d (dx, dy, dw, dh, da) deltas that parameterize the box2box
			transform (see :class:`box_regression.Box2BoxTransformRotated`).

			pred_class_logits: predicted class scores in [-inf, +inf]; use
				softmax(pred_class_logits) to estimate P(class).

			gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
				foreground object classes and K represents the background class.

			pred_proposal_deltas: predicted rotated box2box transform deltas for transforming proposals
				to detection box predictions.

			gt_proposal_deltas: ground-truth rotated box2box transform deltas
	*/

    /**
		Call `fast_rcnn_inference_single_image_rotated` for all images.

		Args:
			boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
				boxes for each image. Element i has shape (Ri, K * 5) if doing
				class-specific regression, or (Ri, 5) if doing class-agnostic
				regression, where Ri is the number of predicted objects for image i.
				This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
			scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
				Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
				for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
			image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
			score_thresh (float): Only return detections with a confidence score exceeding this
				threshold.
			nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
			topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
				all detections.

		Returns:
			instances: (list[Instances]): A list of N instances, one for each image in the batch,
				that stores the topk most confidence detections.
			kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
				the corresponding boxes/scores index in [0, Ri) from the input, for image i.
	*/
	std::tuple<InstancesList, TensorVec> fast_rcnn_inference_rotated(const TensorVec &boxes, const TensorVec &scores,
		const std::vector<ImageSize> &image_shapes, float score_thresh, float nms_thresh, int topk_per_image);

	/**
		Single-image inference. Return rotated bounding-box detection results by thresholding
		on scores and applying rotated non-maximum suppression (Rotated NMS).

		Args:
			Same as `fast_rcnn_inference_rotated`, but with rotated boxes, scores, and image shapes
			per image.

		Returns:
			Same as `fast_rcnn_inference_rotated`, but for only one image.
	*/
	std::tuple<InstancesPtr, torch::Tensor> fast_rcnn_inference_single_image_rotated(
		const torch::Tensor &boxes, const torch::Tensor &scores, const ImageSize &image_shape,
		float score_thresh, float nms_thresh, int topk_per_image);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Two linear layers for predicting Rotated Fast R-CNN outputs.
	class RotatedFastRCNNOutputLayersImpl : public FastRCNNOutputLayersImpl {
	public:
		RotatedFastRCNNOutputLayersImpl(CfgNode &cfg, const ShapeSpec &input_shape);

		/**
		Returns:
			list[Instances]: same as `fast_rcnn_inference_rotated`.
			list[Tensor]: same as `fast_rcnn_inference_rotated`.
		*/
		std::tuple<InstancesList, TensorVec> inference(const TensorVec &predictions, const InstancesList &proposals);
	};
}