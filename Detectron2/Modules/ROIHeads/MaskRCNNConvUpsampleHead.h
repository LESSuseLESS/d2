#pragma once

#include "BaseMaskRCNNHead.h"
#include <Detectron2/Modules/Conv/ConvBn2d.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/roi_heads/mask_head.py

	/**
		A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
		Predictions are made with a final 1x1 conv layer.
	*/
	class MaskRCNNConvUpsampleHeadImpl : public BaseMaskRCNNHeadImpl {
	public:
		/**
			NOTE: this interface is experimental.

			Args:
				input_shape (ShapeSpec): shape of the input feature
				num_classes (int): the number of classes. 1 if using class agnostic prediction.
				conv_dims (list[int]): a list of N>0 integers representing the output dimensions
					of N-1 conv layers and the last upsample layer.
				conv_norm (str or callable): normalization for the conv layers.
					See :func:`detectron2.layers.get_norm` for supported types.
		*/
		MaskRCNNConvUpsampleHeadImpl(CfgNode &cfg, const ShapeSpec &input_shape);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;
		virtual torch::Tensor layers(torch::Tensor x) override;

	private:
		int m_num_classes;
		std::vector<ConvBn2d> m_conv_norm_relus;
		torch::nn::ConvTranspose2d m_deconv{ nullptr };
		ConvBn2d m_predictor{ nullptr };
	};
}