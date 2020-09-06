#pragma once

#include <Detectron2/Structures/Instances.h>
#include <Detectron2/Structures/ShapeSpec.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/roi_heads/keypoint_head.py

	// keypoint heads, which make keypoint predictions from per-region features.
	// Implement the basic Keypoint R-CNN losses and inference logic described in :paper:`Mask R-CNN`.
	class BaseKeypointRCNNHeadImpl : public torch::nn::Module {
	public:
		/**
			Arguments:
				pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
					of instances in the batch, K is the number of keypoints, and S is the side length
					of the keypoint heatmap. The values are spatial logits.
				instances (list[Instances]): A list of M Instances, where M is the batch size.
					These instances are predictions from the model
					that are in 1:1 correspondence with pred_keypoint_logits.
					Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
					instance.
				normalizer (float): Normalize the loss by this amount.
					If not specified, we normalize by the number of visible keypoints in the minibatch.

			Returns a scalar tensor containing the loss.
		*/
		static torch::Tensor keypoint_rcnn_loss(torch::Tensor pred_keypoint_logits,
			const InstancesList &instances, torch::optional<float> normalizer);

		/**
			Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score)
				and add it to the `pred_instances` as a `pred_keypoints` field.

			Args:
				pred_keypoint_logits (Tensor): A tensor of shape (R, K, S, S) where R is the total number
				   of instances in the batch, K is the number of keypoints, and S is the side length of
				   the keypoint heatmap. The values are spatial logits.
				pred_instances (list[Instances]): A list of N Instances, where N is the number of images.

			Returns:
				None. Each element in pred_instances will contain an extra "pred_keypoints" field.
					The field is a tensor of shape (#instance, K, 3) where the last
					dimension corresponds to (x, y, score).
					The scores are larger than 0.
		*/
		static void keypoint_rcnn_inference(const torch::Tensor &pred_keypoint_logits, InstancesList &pred_instances);

	public:
		/**
			num_keypoints (int): number of keypoints to predict
			loss_weight (float): weight to multiple on the keypoint loss
			loss_normalizer (float or str):
				If float, divide the loss by `loss_normalizer * #images`.
				If 'visible', the loss is normalized by the total number of
					visible keypoints across images.
		*/
		BaseKeypointRCNNHeadImpl(CfgNode &cfg);
		virtual ~BaseKeypointRCNNHeadImpl() {}

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
				A dict of losses if in training. The predicted "instances" if in inference.
		*/
		std::tuple<TensorMap, InstancesList> forward(torch::Tensor x, InstancesList &instances);

		// Neural network layers that makes predictions from regional input features.
		virtual torch::Tensor layers(torch::Tensor x) = 0;

	protected:
		int m_num_keypoints;
		float m_loss_weight;
		torch::optional<float> m_loss_normalizer;
	};
	TORCH_MODULE(BaseKeypointRCNNHead);
	using KeypointHead = BaseKeypointRCNNHead;

	// Build a keypoint head from `cfg.MODEL.ROI_KEYPOINT_HEAD.NAME`.
	KeypointHead build_keypoint_head(CfgNode &cfg, const ShapeSpec &input_shape);
}