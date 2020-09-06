#pragma once

#include "MetaArch.h"
#include <Detectron2/Modules/FPN/SemSegFPNHead.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/meta_arch/semantic_seg.py

	// semantic segmentation heads, which make semantic segmentation predictions from feature maps.
	// Main class for semantic segmentation architectures.
	class SemanticSegmentorImpl : public MetaArchImpl {
	public:
		SemanticSegmentorImpl(CfgNode &cfg);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;

		/**
			Args:
				batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
					Each item in the list contains the inputs for one image.

					For now, each item in the list is a dict that contains:

					   * "image": Tensor, image in (C, H, W) format.
					   * "sem_seg": semantic segmentation ground truth
					   * Other information that's included in the original dicts, such as:
						 "height", "width" (int): the output resolution of the model, used in inference.
						 See :meth:`postprocess` for details.

			Returns:
				list[dict]:
				  Each dict is the output for one input image.
				  The dict contains one key "sem_seg" whose value is a
				  Tensor that represents the
				  per-pixel segmentation prediced by the head.
				  The prediction has shape KxHxW that represents the logits of
				  each class for each pixel.
		*/
		virtual std::tuple<InstancesList, TensorMap>
			forward(const std::vector<DatasetMapperOutput> &batched_inputs) override;

	private:
		SemSegFPNHead m_sem_seg_head{ nullptr };
	};
	TORCH_MODULE(SemanticSegmentor);
}