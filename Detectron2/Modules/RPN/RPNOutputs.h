#pragma once

#include <Detectron2/Structures/Box2BoxTransform.h>
#include <Detectron2/Structures/ImageList.h>
#include <Detectron2/Structures/Instances.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/proposal_generator/rpn_outputs.py

	// TODO: comments for future refactoring of this module
	//
	// From @rbg:
	// This code involves a significant amount of tensor reshaping and permuting.Look for
	// ways to simplify this.

	/**
		Shape shorthand in this module:

			N: number of images in the minibatch
			L: number of feature maps per image on which RPN is run
			A: number of cell anchors (must be the same for all feature maps)
			Hi, Wi: height and width of the i-th feature map
			4: size of the box parameterization

		Naming convention:

			objectness: refers to the binary classification of an anchor as object vs. not
			object.

			deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
			transform (see :class:`box_regression.Box2BoxTransform`).

			pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
				sigmoid(pred_objectness_logits) to estimate P(object).

			gt_labels: ground-truth binary classification labels for objectness

			pred_anchor_deltas: predicted box2box transform deltas

			gt_anchor_deltas: ground-truth box2box transform deltas
	*/
	class RPNOutputs {
	public:
		/**
			Args:
				gt_labels (Tensor): shape (N,), each element in {-1, 0, 1} representing
					ground-truth objectness labels with: -1 = ignore; 0 = not object; 1 = object.
				gt_anchor_deltas (Tensor): shape (N, box_dim), row i represents ground-truth
					box2box transform targets (dx, dy, dw, dh) or (dx, dy, dw, dh, da) that map anchor i to
					its matched ground-truth box.
				pred_objectness_logits (Tensor): shape (N,), each element is a predicted objectness
					logit.
				pred_anchor_deltas (Tensor): shape (N, box_dim), each row is a predicted box2box
					transform (dx, dy, dw, dh) or (dx, dy, dw, dh, da)
				smooth_l1_beta (float): The transition point between L1 and L2 loss in
					the smooth L1 loss function. When set to 0, the loss becomes L1. When
					set to +inf, the loss becomes constant 0.

			Returns:
				objectness_loss, localization_loss, both unnormalized (summed over samples).
		*/
		static std::tuple<torch::Tensor, torch::Tensor> rpn_losses(
			const torch::Tensor &gt_labels, const torch::Tensor &gt_anchor_deltas,
			const torch::Tensor &pred_objectness_logits, const torch::Tensor &pred_anchor_deltas,
			float smooth_l1_beta);

	public:
		/**
			box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
				anchor-proposal transformations.
			images (ImageList): :class:`ImageList` instance representing N input images
			batch_size_per_image (int): number of proposals to sample when training
			pred_objectness_logits (list[Tensor]): A list of L elements.
				Element i is a tensor of shape (N, A, Hi, Wi) representing
				the predicted objectness logits for anchors.
			pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
				(N, A*4 or 5, Hi, Wi) representing the predicted "deltas" used to transform anchors
				to proposals.
			anchors (list[Boxes or RotatedBoxes]): A list of Boxes/RotatedBoxes storing the all
				the anchors for each feature map. See :meth:`AnchorGenerator.forward`.
			gt_labels (list[Tensor]): Available on in training.
				See :meth:`RPN.label_and_sample_anchors`.
			gt_boxes (list[Boxes or RotatedBoxes]): Available on in training.
				See :meth:`RPN.label_and_sample_anchors`.
			smooth_l1_beta (float): The transition point between L1 and L2 loss in
				the smooth L1 loss function. When set to 0, the loss becomes L1. When
				set to +inf, the loss becomes constant 0.
		*/
		RPNOutputs(const std::shared_ptr<Box2BoxTransform> &box2box_transform, int batch_size_per_image,
			const ImageList &images, const TensorVec &pred_objectness_logits, const TensorVec &pred_anchor_deltas,
			BoxesList anchors, TensorVec gt_labels, BoxesList gt_boxes, float smooth_l1_beta);

		/**
			Return the losses from a set of RPN predictions and their associated ground-truth.

			Returns:
				dict[loss name -> loss value]: A dict mapping from loss name to loss value.
					Loss names are: `loss_rpn_cls` for objectness classification and
					`loss_rpn_loc` for proposal localization.
		*/
		TensorMap losses();

		/**
			Transform anchors into proposals by applying the predicted anchor deltas.

			Returns:
				proposals (list[Tensor]): A list of L tensors. Tensor i has shape
					(N, Hi*Wi*A, B), where B is box dimension (4 or 5).
		*/
		TensorVec predict_proposals();

		/**
			Return objectness logits in the same format as the proposals returned by
			:meth:`predict_proposals`.

			Returns:
				pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
					(N, Hi*Wi*A).
		*/
		const TensorVec &predict_objectness_logits() {
			return m_pred_objectness_logits;
		}

	private:
		std::shared_ptr<Box2BoxTransform> m_box2box_transform;
		int m_batch_size_per_image;
		TensorVec m_pred_objectness_logits;
		TensorVec m_pred_anchor_deltas;
		BoxesList m_anchors;
		TensorVec m_gt_labels;
		BoxesList m_gt_boxes;
		int m_num_images;
		float m_smooth_l1_beta;
	};
}