#pragma once

#include "StandardROIHeads.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/roi_heads/cascade_rcnn.py

	// Implement :paper:`Cascade R-CNN`.
	class CascadeROIHeadsImpl : public StandardROIHeadsImpl {
	public:
		/**
			NOTE: this interface is experimental.

			Args:
				box_pooler (ROIPooler): pooler that extracts region features from given boxes
				box_heads (list[nn.Module]): box head for each cascade stage
				box_predictors (list[nn.Module]): box predictor for each cascade stage
				proposal_matchers (list[Matcher]): matcher with different IoU thresholds to
					match boxes with ground truth for each stage. The first matcher matches
					RPN proposals with ground truth, the other matchers use boxes predicted
					by the previous stage as proposals and match them with ground truth.
		*/
		CascadeROIHeadsImpl(CfgNode &cfg) : StandardROIHeadsImpl(cfg) {}
		CascadeROIHeadsImpl(CfgNode &cfg, std::vector<BoxHead> box_heads,
			std::vector<FastRCNNOutputLayers> box_predictors, std::vector<std::shared_ptr<Matcher>> proposal_matchers);

		virtual std::tuple<InstancesList, TensorMap> forward(const ImageList &images, const TensorMap &features,
			InstancesList &proposals, const InstancesList &targets = {}) override;

	private:
		virtual void _init_box_head(CfgNode &cfg, const ShapeSpec::Map &input_shapes) override;

		/**
			Args:
				features, targets: the same as in
					Same as in :meth:`ROIHeads.forward`.
				proposals (list[Instances]): the per-image object proposals with
					their matching ground truth.
					Each has fields "proposal_boxes", and "objectness_logits",
					"gt_classes", "gt_boxes".
		*/
		std::tuple<TensorMap, InstancesList> _forward_box(const TensorMap &features, InstancesList &proposals,
			const InstancesList &targets = {});

		/**
			Match proposals with groundtruth using the matcher at the given stage.
			Label the proposals as foreground or background based on the match.

			Args:
				proposals (list[Instances]): One Instances for each image, with
					the field "proposal_boxes".
				stage (int): the current stage
				targets (list[Instances]): the ground truth instances

			Returns:
				list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
		*/
		InstancesList _match_and_label_boxes(InstancesList &proposals, int stage, const InstancesList &targets);

		/**
			Args:
				features (list[Tensor]): #lvl input features to ROIHeads
				proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
				stage (int): the current stage

			Returns:
				Same output as `FastRCNNOutputLayers.forward()`.
		*/
		TensorVec _run_stage(const TensorVec &features, InstancesList &proposals, int stage);

		/**
			Args:
				boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
				image_sizes (list[tuple]): list of image shapes in (h, w)

			Returns:
				list[Instances]: per-image proposals with the given boxes.
		*/
		InstancesList _create_proposals_from_boxes(const TensorVec &boxes, const std::vector<ImageSize> &image_sizes);

	private:
		int m_num_cascade_stages;
		std::vector<std::shared_ptr<Matcher>> m_proposal_matchers;
	};
}
