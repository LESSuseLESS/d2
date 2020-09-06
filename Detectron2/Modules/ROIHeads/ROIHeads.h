#pragma once

#include <Detectron2/Structures/ImageList.h>
#include <Detectron2/Structures/Instances.h>
#include <Detectron2/Structures/Matcher.h>
#include <Detectron2/Structures/ShapeSpec.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/roi_heads/roi_heads.py

	// ROIHeads take feature maps and region proposals, and perform per-region computation.
	/**
		ROIHeads perform all per-region computation in an R-CNN.

		It typically contains logic to
		1. (in training only) match proposals with ground truth and sample them
		2. crop the regions and extract per-region features using proposals
		3. make per-region predictions with different heads

		It can have many variants, implemented as subclasses of this class.
		This base class contains the logic to match/sample proposals.
		But it is not necessary to inherit this class if the sampling logic is not needed.
	 */
	class ROIHeadsImpl : public torch::nn::Module {
	public:
		/**
			Given a list of N Instances (for N images), each containing a `gt_classes` field,
			return a list of Instances that contain only instances with `gt_classes != -1 &&
			gt_classes != bg_label`.

			Args:
				proposals (list[Instances]): A list of N Instances, where N is the number of
					images in the batch.
				bg_label: label index of background class.

			Returns:
				list[Instances]: N Instances, each contains only the selected foreground instances.
				list[Tensor]: N boolean vector, correspond to the selection mask of
					each Instances object. True for selected instances.
		*/
		static std::tuple<InstancesList, TensorVec>
			select_foreground_proposals(const InstancesList &proposals, int bg_label);

		/**
			Args:
				proposals (list[Instances]): a list of N Instances, where N is the
					number of images.

			Returns:
				proposals: only contains proposals with at least one visible keypoint.

			Note that this is still slightly different from Detectron.
			In Detectron, proposals for training keypoint head are re-sampled from
			all the proposals with IOU>threshold & >=1 visible keypoint.

			Here, the proposals are first sampled from all proposals with
			IOU>threshold, then proposals with no visible keypoint are filtered out.
			This strategy seems to make no difference on Detectron and is easier to implement.
		*/
		static InstancesList select_proposals_with_visible_keypoints(const InstancesList &proposals);

		static void update(TensorMap &dest, const TensorMap &src);

	public:
		ROIHeadsImpl(CfgNode &cfg);
		virtual ~ROIHeadsImpl() {}

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) = 0;

		/**
			Args:
				features (dict[str,Tensor]): input data as a mapping from feature
					map name to tensor. Axis 0 represents the number of images `N` in
					the input data; axes 1-3 are channels, height, and width, which may
					vary between feature maps (e.g., if a feature pyramid is used).
				proposals (list[Instances]): length `N` list of `Instances`. The i-th
					`Instances` contains object proposals for the i-th input image,
					with fields "proposal_boxes" and "objectness_logits".
				targets (list[Instances], optional): length `N` list of `Instances`. The i-th
					`Instances` contains the ground-truth per-instance annotations
					for the i-th input image.  Specify `targets` during training only.
					It may have the following fields:

					- gt_boxes: the bounding box of each instance.
					- gt_classes: the label for each instance with a category ranging in [0, #class].
					- gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
					- gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

			Returns:
				list[Instances]: length `N` list of `Instances` containing the
				detected instances. Returned during inference only; may be [] during training.

				dict[str->Tensor]:
				mapping from a named loss to a tensor storing the loss. Used during training only.
		*/
		virtual std::tuple<InstancesList, TensorMap> forward(const ImageList &images, const TensorMap &features,
			InstancesList &proposals, const InstancesList &targets = {}) = 0;

		virtual InstancesList forward_with_given_boxes(const TensorMap &features, InstancesList &instances) = 0;

	protected:
		int m_num_classes;					// number of classes. Used to label background proposals.
		int m_batch_size_per_image;			// number of proposals to use for training
		float m_positive_sample_fraction;	// fraction of positive (foreground) proposals to use for training.
		Matcher m_proposal_matcher;			// matcher that matches proposals and ground truth
		bool m_proposal_append_gt;			// whether to include ground truth as proposals as well

		/**
			Augment `proposals` with ground-truth boxes from `gt_boxes`.

			Args:
				Same as `add_ground_truth_to_proposals`, but with gt_boxes and proposals
				per image.

			Returns:
				Same as `add_ground_truth_to_proposals`, but for only one image.
		*/
		InstancesPtr add_ground_truth_to_proposals_single_image(const torch::Tensor &gt_boxes,
			InstancesPtr proposals);

		/**
			Call `add_ground_truth_to_proposals_single_image` for all images.

			Args:
				gt_boxes(list[Boxes]): list of N elements. Element i is a Boxes
					representing the gound-truth for image i.
				proposals (list[Instances]): list of N elements. Element i is a Instances
					representing the proposals for image i.

			Returns:
				list[Instances]: list of N Instances. Each is the proposals for the image,
					with field "proposal_boxes" and "objectness_logits".
		*/
		void add_ground_truth_to_proposals(InstancesList &proposals, const BoxesList &gt_boxes);

		/**
			Based on the matching between N proposals and M groundtruth,
			sample the proposals and set their classification labels.

			Args:
				matched_idxs (Tensor): a vector of length N, each is the best-matched
					gt index in [0, M) for each proposal.
				matched_labels (Tensor): a vector of length N, the matcher's label
					(one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
				gt_classes (Tensor): a vector of length M.

			Returns:
				Tensor: a vector of indices of sampled proposals. Each is in [0, N).
				Tensor: a vector of the same length, the classification label for
					each sampled proposal. Each sample is labeled as either a category in
					[0, num_classes) or the background (num_classes).
		*/
		std::tuple<torch::Tensor, torch::Tensor> _sample_proposals(const torch::Tensor &matched_idxs,
			const torch::Tensor &matched_labels, torch::Tensor gt_classes);

		/**
			Prepare some proposals to be used to train the ROI heads.
			It performs box matching between `proposals` and `targets`, and assigns
			training labels to the proposals.
			It returns ``m_batch_size_per_image`` random samples from proposals and groundtruth
			boxes, with a fraction of positives that is no larger than
			``m_positive_sample_fraction``.

			Args:
				See :meth:`ROIHeads.forward`

			Returns:
				list[Instances]:
					length `N` list of `Instances`s containing the proposals
					sampled for training. Each `Instances` has the following fields:

					- proposal_boxes: the proposal boxes
					- gt_boxes: the ground-truth box that the proposal is assigned to
					  (this is only meaningful if the proposal has a label > 0; if label = 0
					  then the ground-truth box is random)

					Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
		*/
		InstancesList label_and_sample_proposals(InstancesList &proposals, const InstancesList &targets);
	};
	TORCH_MODULE(ROIHeads);

	// Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
	ROIHeads build_roi_heads(CfgNode &cfg, const ShapeSpec::Map &input_shapes);
}