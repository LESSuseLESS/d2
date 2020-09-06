#pragma once

#include "MetaArch.h"
#include <Detectron2/Structures/PanopticSegment.h>
#include <Detectron2/Modules/FPN/SemSegFPNHead.h>
#include <Detectron2/Modules/ROIHeads/ROIHeads.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/meta_arch/panoptic_fpn.py

	// Implement the paper :paper:`PanopticFPN`.
	class PanopticFPNImpl : public MetaArchImpl {
	public:
		PanopticFPNImpl(CfgNode &cfg);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;

		/**
			Args:
				batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
					Each item in the list contains the inputs for one image.

					For now, each item in the list is a dict that contains:

					* "image": Tensor, image in (C, H, W) format.
					* "instances": Instances
					* "sem_seg": semantic segmentation ground truth.
					* Other information that's included in the original dicts, such as:
					  "height", "width" (int): the output resolution of the model, used in inference.
					  See :meth:`postprocess` for details.

			Returns:
				list[dict]:
					each dict is the results for one image. The dict contains the following keys:

					* "instances": see :meth:`GeneralizedRCNN.forward` for its format.
					* "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
					* "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
					  See the return value of
					  :func:`combine_semantic_and_instance_outputs` for its format.
        */
		virtual std::tuple<InstancesList, TensorMap>
			forward(const std::vector<DatasetMapperOutput> &batched_inputs) override;

	private:
		SemSegFPNHead m_sem_seg_head{ nullptr };
		ROIHeads m_roi_heads{ nullptr };

		float m_instance_loss_weight;

		// options when combining instance & semantic outputs
		bool m_combine_on;
		float m_combine_overlap_threshold;
		float m_combine_stuff_area_limit;
		float m_combine_instances_confidence_threshold;

		/**
			Implement a simple combining logic following
			"combine_semantic_and_instance_predictions.py" in panopticapi
			to produce panoptic segmentation outputs.

			Args:
				instance_results: output of :func:`detector_postprocess`.
				semantic_results: an (H, W) tensor, each is the contiguous semantic
					category id

			Returns:
				panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
				segments_info (list[dict]): Describe each segment in `panoptic_seg`.
					Each dict contains keys "id", "category_id", "isthing".
		*/
		std::shared_ptr<PanopticSegment> combine_semantic_and_instance_outputs(
			const InstancesPtr &instance_results, const torch::Tensor &semantic_results);
	};
	TORCH_MODULE(PanopticFPN);
}