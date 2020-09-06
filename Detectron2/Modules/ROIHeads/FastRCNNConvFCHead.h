#pragma once

#include <Detectron2/Modules/Conv/ConvBn2d.h>
#include <Detectron2/Structures/ShapeSpec.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/roi_heads/box_head.py

	// FastRCNNConvFCHead: makes box predictions from per-region features.
	class FastRCNNConvFCHeadImpl : public torch::nn::Module {
	public:
		// input_shape: shape of the input feature.
		FastRCNNConvFCHeadImpl(CfgNode &cfg, const ShapeSpec &input_shape);
		void initialize(const ModelImporter &importer, const std::string &prefix);

		// ShapeSpec: the output feature shape
		ShapeSpec output_shape() const {
			return m_output_size;
		}

		torch::Tensor forward(torch::Tensor x);

	private:
		ShapeSpec m_output_size;
		std::vector<ConvBn2d> m_conv_norm_relus;
		std::vector<torch::nn::Linear> m_fcs;
	};
	TORCH_MODULE(FastRCNNConvFCHead);
	using BoxHead = FastRCNNConvFCHead;

	//  Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
	BoxHead build_box_head(CfgNode &cfg, const ShapeSpec &input_shape);
}