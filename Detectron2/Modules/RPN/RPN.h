#pragma once

#include <Detectron2/Structures/Box2BoxTransform.h>
#include <Detectron2/Structures/ImageList.h>
#include <Detectron2/Structures/Instances.h>
#include <Detectron2/Structures/Matcher.h>
#include <Detectron2/Structures/ShapeSpec.h>
#include "StandardRPNHead.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/proposal_generator/rpn.py

	// Region Proposal Network (RPN), introduced by :paper:`Faster R-CNN`.
	class RPNImpl : public torch::nn::Module {
	public:
		RPNImpl(CfgNode &cfg, const ShapeSpec::Map &input_shapes);
		virtual ~RPNImpl() {}

		void initialize(const ModelImporter &importer, const std::string &prefix);

		/*
			Args:
				images (ImageList): input images of length `N`
				features (dict[str: Tensor]): input data as a mapping from feature
					map name to tensor. Axis 0 represents the number of images `N` in
					the input data; axes 1-3 are channels, height, and width, which may
					vary between feature maps (e.g., if a feature pyramid is used).
				gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
					Each `Instances` stores ground-truth instances for the corresponding image.

			Returns:
				proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
				loss: dict[Tensor] or None
		 */
		std::tuple<InstancesList, TensorMap> forward(const ImageList &images, const TensorMap &features_,
			const InstancesList &gt_instances = {});

	protected:
		std::vector<std::string> m_in_features; // = cfg.MODEL.RPN.IN_FEATURES

		AnchorGenerator m_anchor_generator{ nullptr };
		std::shared_ptr<Box2BoxTransform> m_box2box_transform;
		Matcher m_anchor_matcher;
		RPNHead m_rpn_head{ nullptr };

		int m_min_box_side_len;
		float m_nms_thresh;
		int m_batch_size_per_image;
		float m_positive_fraction;
		float m_smooth_l1_beta;
		float m_loss_weight;
		int m_pre_nms_topk[2];
		int m_post_nms_topk[2];
		int m_boundary_threshold;

		/**
			Randomly sample a subset of positive and negative examples, and overwrite
			the label vector to the ignore value (-1) for all elements that are not
			included in the sample.

			Args:
				labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
		*/
		torch::Tensor _subsample_labels(torch::Tensor label);

		/**
			Args:
				anchors (list[Boxes]): anchors for each feature map.
				gt_instances: the ground-truth instances for each image.

			Returns:
				list[Tensor]:
					List of #img tensors. i-th element is a vector of labels whose length is
					the total number of anchors across feature maps. Label values are in {-1, 0, 1},
					with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
				list[Tensor]:
					i-th element is a Nx4 tensor, where N is the total number of anchors across
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
				proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
					All proposal predictions on the feature maps.
				pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
				images (ImageList): Input images as an :class:`ImageList`.
				nms_thresh (float): IoU threshold to use for NMS
				pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
					When RPN is run on multiple feature maps (as in FPN) this number is per
					feature map.
				post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
					When RPN is run on multiple feature maps (as in FPN) this number is total,
					over all feature maps.
				min_box_side_len (float): minimum proposal box side length in pixels (absolute units
					wrt input images).
				training (bool): True if proposals are to be used in training, otherwise False.
					This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
					comment.

			Returns:
				proposals (list[Instances]): list of N Instances. The i-th Instances
					stores post_nms_topk object proposals for image i, sorted by their
					objectness score in descending order.
		*/
		// made these two virtual to each other: find_top_rpn_proposals(), find_top_rrpn_proposals()
		virtual InstancesList find_top_proposals(const TensorVec &proposals,
			const TensorVec &pred_objectness_logits, const ImageList &images, float nms_thresh,
			int pre_nms_topk, int post_nms_topk, float min_box_side_len, bool training);
	};
	TORCH_MODULE(RPN);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/proposal_generator/build.py

	/**
		Build a proposal generator from `cfg.MODEL.PROPOSAL_GENERATOR.NAME`.
		The name can be "PrecomputedProposals" to use no proposal generator.
	*/
	RPN build_proposal_generator(CfgNode &cfg, const ShapeSpec::Map &input_shapes);
}