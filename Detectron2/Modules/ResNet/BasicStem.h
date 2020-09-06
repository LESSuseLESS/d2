#pragma once

#include "CNNBlockBase.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/resnet.py

	// The standard ResNet stem (layers before the first residual block).
	class BasicStemImpl : public CNNBlockBaseImpl {
	public:
		// norm (str or callable): norm after the first conv layer.
		//   See : func:`layers.get_norm` for supported format.
		BasicStemImpl(int in_channels, int out_channels, BatchNorm::Type norm);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;
		virtual torch::Tensor forward(torch::Tensor x) override;

	private:
		int m_in_channels;

		ConvBn2d m_convbn1;
	};
	TORCH_MODULE(BasicStem);
}