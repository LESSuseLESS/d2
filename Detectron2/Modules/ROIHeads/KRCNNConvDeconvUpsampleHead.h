#pragma once

#include "BaseKeypointRCNNHead.h"
#include <Detectron2/Modules/Conv/ConvBn2d.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/roi_heads/keypoint_head.py

	/**
		A standard keypoint head containing a series of 3x3 convs, followed by
		a transpose convolution and bilinear interpolation for upsampling.
	*/
	class KRCNNConvDeconvUpsampleHeadImpl : public BaseKeypointRCNNHeadImpl {
	public:
		/**
			NOTE: this interface is experimental.

			Args:
				input_shape (ShapeSpec): shape of the input feature
				conv_dims: an iterable of output channel counts for each conv in the head
							 e.g. (512, 512, 512) for three convs outputting 512 channels.
		*/
		KRCNNConvDeconvUpsampleHeadImpl(CfgNode &cfg, const ShapeSpec &input_shape);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;
		virtual torch::Tensor layers(torch::Tensor x) override;

	private:
		int m_up_scale;
		std::vector<ConvBn2d> m_blocks;
		torch::nn::ConvTranspose2d m_score_lowres{ nullptr };
	};
}