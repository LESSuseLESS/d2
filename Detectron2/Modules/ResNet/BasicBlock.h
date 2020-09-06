#pragma once

#include "CNNBlockBase.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/resnet.py

	/**
		The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
		with two 3x3 conv layers and a projection shortcut if needed.
	*/
	class BasicBlockImpl : public CNNBlockBaseImpl {
	public:
		/**
			in_channels (int): Number of input channels.
			out_channels (int): Number of output channels.
			stride (int): Stride for the first conv.
			norm (str or callable): normalization for all conv layers.
				See :func:`layers.get_norm` for supported format.
		*/
		BasicBlockImpl(int in_channels, int out_channels, int stride, BatchNorm::Type norm);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;
		virtual torch::Tensor forward(torch::Tensor x) override;

	private:
		ConvBn2d m_shortcut{ nullptr };
		ConvBn2d m_convbn1;
		ConvBn2d m_convbn2;
	};
}