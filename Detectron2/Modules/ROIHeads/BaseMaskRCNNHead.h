#pragma once

#include <Detectron2/Structures/Instances.h>
#include <Detectron2/Structures/ShapeSpec.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/roi_heads/mask_head.py
	
	// mask heads, which predicts instance masks given
	// Implement the basic Mask R-CNN losses and inference logic described in :paper:`Mask R-CNN`
	class BaseMaskRCNNHeadImpl : public torch::nn::Module {
	public:
		/**
			Compute the mask prediction loss defined in the Mask R-CNN paper.

			Args:
				pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
					for class-specific or class-agnostic, where B is the total number of predicted masks
					in all images, C is the number of foreground classes, and Hmask, Wmask are the height
					and width of the mask predictions. The values are logits.
				instances (list[Instances]): A list of N Instances, where N is the number of images
					in the batch. These instances are in 1:1
					correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
					...) associated with each instance are stored in fields.
				vis_period (int): the period (in steps) to dump visualization.

			Returns:
				mask_loss (Tensor): A scalar tensor containing the loss.
		*/
		static torch::Tensor mask_rcnn_loss(torch::Tensor pred_mask_logits,
			const InstancesList &instances, int vis_period = 0);

		/**
			Convert pred_mask_logits to estimated foreground probability masks while also
			extracting only the masks for the predicted classes in pred_instances. For each
			predicted box, the mask of the same class is attached to the instance by adding a
			new "pred_masks" field to pred_instances.

			Args:
				pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
					for class-specific or class-agnostic, where B is the total number of predicted masks
					in all images, C is the number of foreground classes, and Hmask, Wmask are the height
					and width of the mask predictions. The values are logits.
				pred_instances (list[Instances]): A list of N Instances, where N is the number of images
					in the batch. Each Instances must have field "pred_classes".

			Returns:
				None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
					Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
					masks the resolution predicted by the network; post-processing steps, such as resizing
					the predicted masks to the original image resolution and/or binarizing them, is left
					to the caller.
		*/
		static void mask_rcnn_inference(const torch::Tensor &pred_mask_logits, InstancesList &pred_instances);

	public:
		// vis_period (int): visualization period
		BaseMaskRCNNHeadImpl(CfgNode &cfg);
		virtual ~BaseMaskRCNNHeadImpl() {}

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) = 0;

		/**
			Args:
				x: input region feature(s) provided by :class:`ROIHeads`.
				instances (list[Instances]): contains the boxes & labels corresponding
					to the input features.
					Exact format is up to its caller to decide.
					Typically, this is the foreground instances in training, with
					"proposal_boxes" field and other gt annotations.
					In inference, it contains boxes that are already predicted.

			Returns:
				A dict of losses in training. The predicted "instances" in inference.
		*/
		std::tuple<TensorMap, InstancesList> forward(torch::Tensor x, InstancesList &instances);

		// Neural network layers that makes predictions from input features.
		virtual torch::Tensor layers(torch::Tensor x) = 0;

	private:
		int m_vis_period;
	};
	TORCH_MODULE(BaseMaskRCNNHead);
	using MaskHead = BaseMaskRCNNHead;

	// Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
	MaskHead build_mask_head(CfgNode &cfg, const ShapeSpec &input_shape);
}