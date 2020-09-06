#pragma once

#include "MetaArch.h"
#include <Detectron2/Modules/ROIHeads/ROIHeads.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/meta_arch/rcnn.py

	// Generalized R-CNN. Any models that contains the following three components:
    //	1. Per-image feature extraction (aka backbone)
    //	2. Region proposal generation
    //	3. Per-region feature extraction and prediction

	class GeneralizedRCNNImpl : public MetaArchImpl {
	public:
		GeneralizedRCNNImpl(CfgNode &cfg);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;

		/**
			Args:
				batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
					Each item in the list contains the inputs for one image.
					For now, each item in the list is a dict that contains:

					* image: Tensor, image in (C, H, W) format.
					* instances (optional): groundtruth :class:`Instances`
					* proposals (optional): :class:`Instances`, precomputed proposals.

					Other information that's included in the original dicts, such as:

					* "height", "width" (int): the output resolution of the model, used in inference.
						See :meth:`postprocess` for details.

			Returns:
				list[dict]:
					Each dict is the output for one input image.
					The dict contains one key "instances" whose value is a :class:`Instances`.
					The :class:`Instances` object has the following keys:
					"pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
		*/
		virtual std::tuple<InstancesList, TensorMap>
			forward(const std::vector<DatasetMapperOutput> &batched_inputs) override;

		/**
			Run inference on the given inputs.

			Args:
				batched_inputs (list[dict]): same as in :meth:`forward`
				detected_instances (None or list[Instances]): if not None, it
					contains an `Instances` object per image. The `Instances`
					object contains "pred_boxes" and "pred_classes" which are
					known boxes in the image.
					The inference will then skip the detection of bounding boxes,
					and only predict other per-ROI outputs.
				do_postprocess (bool): whether to apply post-processing on the outputs.

			Returns:
				same as in :meth:`forward`.
		*/
		InstancesList inference(const std::vector<DatasetMapperOutput> &batched_inputs,
			const InstancesList &detected_instances = {}, bool do_postprocess = true);

		/**
			A function used to visualize images and proposals. It shows ground truth
			bounding boxes on the original image and up to 20 predicted object
			proposals on the original image. Users can implement different
			visualization functions for different models.

			Args:
				batched_inputs (list): a list that contains input to the model.
				proposals (list): a list that contains predicted proposals. Both
					batched_inputs and proposals should have the same length.
		*/
		void visualize_training(const std::vector<DatasetMapperOutput> &batched_inputs,
			const InstancesList &proposals);

	private:
		ROIHeads m_roi_heads{ nullptr };
	};
	TORCH_MODULE(GeneralizedRCNN);
}