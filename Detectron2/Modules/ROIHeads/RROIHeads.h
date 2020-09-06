#pragma once

#include "StandardROIHeads.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/roi_heads/rotated_fast_rcnn.py

	/**
		This class is used by Rotated Fast R-CNN to detect rotated boxes.
		For now, it only supports box predictions but not mask or keypoints.
	*/
	class RROIHeadsImpl : public StandardROIHeadsImpl {
	public:
		RROIHeadsImpl(CfgNode &cfg);
		void Create(CfgNode &cfg, const ShapeSpec::Map &input_shapes);

		/**
			Prepare some proposals to be used to train the RROI heads.
			It performs box matching between `proposals` and `targets`, and assigns
			training labels to the proposals.
			It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
			with a fraction of positives that is no larger than `self.positive_sample_fraction.

			Args:
				See :meth:`StandardROIHeads.forward`

			Returns:
				list[Instances]: length `N` list of `Instances`s containing the proposals
					sampled for training. Each `Instances` has the following fields:
					- proposal_boxes: the rotated proposal boxes
					- gt_boxes: the ground-truth rotated boxes that the proposal is assigned to
					  (this is only meaningful if the proposal has a label > 0; if label = 0
					   then the ground-truth box is random)
					- gt_classes: the ground-truth classification lable for each proposal
		*/
		InstancesList label_and_sample_proposals(InstancesList &proposals, const InstancesList &targets);

	private:
		virtual void _init_box_head(CfgNode &cfg, const ShapeSpec::Map &input_shapes) override;
	};
}
