#pragma once

#include <Detectron2/Modules/ROIPooler/ROIPooler.h>
#include "ROIHeads.h"
#include "FastRCNNOutputLayers.h"
#include "BaseMaskRCNNHead.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/roi_heads/roi_heads.py

	// The ROIHeads in a typical "C4" R-CNN model, where the box and mask head share the cropping and the per-region
	// feature computation by a Res5 block.
	class Res5ROIHeadsImpl : public ROIHeadsImpl {
	public:
		Res5ROIHeadsImpl(CfgNode &cfg, const ShapeSpec::Map &input_shapes);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;

		virtual std::tuple<InstancesList, TensorMap> forward(const ImageList &images, const TensorMap &features,
			InstancesList &proposals, const InstancesList &targets = {}) override;

		/**
			Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

			Args:
				features: same as in `forward()`
				instances (list[Instances]): instances to predict other outputs. Expect the keys
					"pred_boxes" and "pred_classes" to exist.

			Returns:
				instances (Instances):
					the same `Instances` object, with extra
					fields such as `pred_masks` or `pred_keypoints`.
		*/
		virtual InstancesList forward_with_given_boxes(const TensorMap &features, InstancesList &instances) override;

	private:
		std::vector<std::string> m_in_features;
		bool m_mask_on;
		ROIPooler m_pooler{ nullptr };
		FastRCNNOutputLayers m_box_predictor{ nullptr };
		torch::nn::Sequential m_res5;
		MaskHead m_mask_head{ nullptr };

		int _build_res5_block(CfgNode &cfg);
		torch::Tensor _shared_roi_transform(const TensorVec &features, const BoxesList &boxes);
		TensorVec select_features(const TensorMap &features);
	};
	TORCH_MODULE(Res5ROIHeads);
}