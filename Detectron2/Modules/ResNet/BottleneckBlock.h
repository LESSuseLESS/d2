#pragma once

#include "CNNBlockBase.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/resnet.py

	/**
		The standard bottleneck residual block used by ResNet-50, 101 and 152
		defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
		1x1, 3x3, 1x1, and a projection shortcut if needed.
	*/
	class BottleneckBlockImpl : public CNNBlockBaseImpl {
	public:
		/**
			bottleneck_channels (int): number of output channels for the 3x3
				"bottleneck" conv layers.
			num_groups (int): number of groups for the 3x3 conv layer.
			norm (str or callable): normalization for all conv layers.
				See :func:`layers.get_norm` for supported format.
			stride_in_1x1 (bool): when stride>1, whether to put stride in the
				first 1x1 convolution or the bottleneck 3x3 convolution.
			dilation (int): the dilation rate of the 3x3 conv layer.
		*/
		BottleneckBlockImpl(int in_channels, int out_channels, int bottleneck_channels, int stride,
			int num_groups, BatchNorm::Type norm, bool stride_in_1x1, int dilation = 1);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;
		virtual torch::Tensor forward(torch::Tensor x) override;

	private:
		ConvBn2d m_shortcut{ nullptr };
		ConvBn2d m_convbn1;
		ConvBn2d m_convbn2;
		ConvBn2d m_convbn3;
	};
}