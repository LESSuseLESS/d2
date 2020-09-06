#pragma once

#include "RPN.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/proposal_generator/rrpn.py

	// Rotated Region Proposal Network described in :paper:`RRPN`.
	class RRPNImpl : public RPNImpl {
	public:
		RRPNImpl(CfgNode &cfg, const ShapeSpec::Map &input_shapes);

	private:
		/**
		Args:
			anchors (list[RotatedBoxes]): anchors for each feature map.
			gt_instances: the ground-truth instances for each image.

		Returns:
			list[Tensor]:
				List of #img tensors. i-th element is a vector of labels whose length is
				the total number of anchors across feature maps. Label values are in {-1, 0, 1},
				with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
			list[Tensor]:
				i-th element is a Nx5 tensor, where N is the total number of anchors across
				feature maps.  The values are the matched gt boxes for each anchor.
				Values are undefined for those anchors not labeled as 1.
		*/
		std::tuple<TensorVec, BoxesList> label_and_sample_anchors(const BoxesList &anchors,
			const InstancesList &gt_instances);

		/**
			For each feature map, select the `pre_nms_topk` highest scoring proposals,
			apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
			highest scoring proposals among all the feature maps if `training` is True,
			otherwise, returns the highest `post_nms_topk` scoring proposals for each
			feature map.

			Args:
				proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 5).
					All proposal predictions on the feature maps.
				pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
				images (ImageList): Input images as an :class:`ImageList`.
				nms_thresh (float): IoU threshold to use for NMS
				pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
					When RRPN is run on multiple feature maps (as in FPN) this number is per
					feature map.
				post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
					When RRPN is run on multiple feature maps (as in FPN) this number is total,
					over all feature maps.
				min_box_side_len (float): minimum proposal box side length in pixels (absolute units
					wrt input images).
				training (bool): True if proposals are to be used in training, otherwise False.
					This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
					comment.

			Returns:
				proposals (list[Instances]): list of N Instances. The i-th Instances
					stores post_nms_topk object proposals for image i.
		*/
		virtual InstancesList find_top_proposals(const TensorVec &proposals,
			const TensorVec &pred_objectness_logits, const ImageList &images, float nms_thresh,
			int pre_nms_topk, int post_nms_topk, float min_box_side_len, bool training) override;
	};
}