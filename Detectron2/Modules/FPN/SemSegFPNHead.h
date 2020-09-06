#pragma once

#include <Detectron2/Structures/Instances.h>
#include <Detectron2/Structures/ShapeSpec.h>
#include <Detectron2/Modules/Conv/ConvBn2d.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/meta_arch/semantic_seg.py

	/**
		A semantic segmentation head described in :paper:`PanopticFPN`.
		It takes FPN features as input and merges information from all
		levels of the FPN into single output.
	*/
	class SemSegFPNHeadImpl : public torch::nn::Module {
	public:
		SemSegFPNHeadImpl(CfgNode &cfg, const ShapeSpec::Map &input_shapes);

		void initialize(const ModelImporter &importer, const std::string &prefix);

		int ignore_value() const { return m_ignore_value; }

		/**
			Returns:
				In training, returns (None, dict of losses)
				In inference, returns (CxHxW logits, {})
		*/
		std::tuple<torch::Tensor, TensorMap> forward(const TensorMap &features, const torch::Tensor &targets);

	private:
		std::vector<std::string> m_in_features;
		int m_ignore_value;		// Label in the semantic segmentation ground truth that is ignored, i.e., no loss is
								// calculated for the correposnding pixel.
		int m_common_stride;	// Outputs from semantic - FPN heads are up - scaled to the COMMON_STRIDE stride.
		float m_loss_weight;
		torch::nn::functional::InterpolateFuncOptions m_interpolate_options;

		std::vector<torch::nn::Sequential> m_scale_heads;
		ConvBn2d m_predictor{ nullptr };

		torch::Tensor layers(const TensorMap &features);
		TensorMap losses(torch::Tensor predictions, torch::Tensor targets);
	};
	TORCH_MODULE(SemSegFPNHead);
}