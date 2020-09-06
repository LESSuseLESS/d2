#pragma once

#include <Detectron2/Structures/Box2BoxTransform.h>
#include <Detectron2/Structures/Instances.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/roi_heads/fast_rcnn.py

	/**
		Shape shorthand in this module:

			N: number of images in the minibatch
			R: number of ROIs, combined over all images, in the minibatch
			Ri: number of ROIs in image i
			K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

		Naming convention:

			deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
			transform (see :class:`box_regression.Box2BoxTransform`).

			pred_class_logits: predicted class scores in [-inf, +inf]; use
				softmax(pred_class_logits) to estimate P(class).

			gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
				foreground object classes and K represents the background class.

			pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
				to detection box predictions.

			gt_proposal_deltas: ground-truth box2box transform deltas
	*/

    /**
		A class that stores information about outputs of a Fast R-CNN head.
		It provides methods that are used to decode the outputs of a Fast R-CNN head.
	*/
	class FastRCNNOutputs {
	public:
		/**
			Args:
				box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
					box2box transform instance for proposal-to-detection transformations.
				pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
					logits for all R predicted object instances.
					Each row corresponds to a predicted object instance.
				pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
					class-specific or class-agnostic regression. It stores the predicted deltas that
					transform proposals into final box detections.
					B is the box dimension (4 or 5).
					When B is 4, each row is [dx, dy, dw, dh (, ....)].
					When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
				proposals (list[Instances]): A list of N Instances, where Instances i stores the
					proposals for image i, in the field "proposal_boxes".
					When training, each Instances must have ground-truth labels
					stored in the field "gt_classes" and "gt_boxes".
					The total number of all instances must be equal to R.
				smooth_l1_beta (float): The transition point between L1 and L2 loss in
					the smooth L1 loss function. When set to 0, the loss becomes L1. When
					set to +inf, the loss becomes constant 0.
		*/
		FastRCNNOutputs(const std::shared_ptr<Box2BoxTransform> &box2box_transform,
			const torch::Tensor &pred_class_logits, const torch::Tensor &pred_proposal_deltas,
			const InstancesList &proposals, float smooth_l1_beta = 0);

		/**
			Compute the softmax cross entropy loss for box classification.

			Returns:
				scalar Tensor
		*/
		torch::Tensor softmax_cross_entropy_loss();

		/**
			Compute the smooth L1 loss for box regression.

			Returns:
				scalar Tensor
		*/
		torch::Tensor smooth_l1_loss();

		/**
			A subclass is expected to have the following methods because
			they are used to query information about the head predictions.
		*/

		/**
			Compute the default losses for box head in Fast(er) R-CNN,
			with softmax cross entropy loss and smooth L1 loss.

			Returns:
				A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
		*/
		TensorMap losses();

		// Deprecated
		TensorVec predict_boxes();

		// Deprecated
		TensorVec predict_probs();

		// Deprecated
		std::tuple<InstancesList, TensorVec> inference(float score_thresh, float nms_thresh, int topk_per_image);

	private:
		std::shared_ptr<Box2BoxTransform> m_box2box_transform;
		std::vector<int64_t> m_num_preds_per_image;
		torch::Tensor m_pred_class_logits;
		torch::Tensor m_pred_proposal_deltas;
		float m_smooth_l1_beta;
		std::vector<ImageSize> m_image_shapes;

		torch::Tensor m_proposals;
		torch::Tensor m_gt_boxes;
		torch::Tensor m_gt_classes;
		bool m_no_instances;

		// Log the accuracy metrics to EventStorage.
		void _log_accuracy();

		/**
			Returns:
				Tensor: A Tensors of predicted class-specific or class-agnostic boxes
					for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
					the number of predicted objects for image i and B is the box dimension (4 or 5)
		*/
		torch::Tensor _predict_boxes();
	};
}