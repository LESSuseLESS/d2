#pragma once

#include <Detectron2/Modules/BatchNorm/BatchNorm.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from layers/wrappers.py

	// A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
	class ConvBn2dImpl : public torch::nn::Module {
	public:
		/**
			Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

			Args:
				norm (nn.Module, optional): a normalization layer
				activation (callable(Tensor) -> Tensor): a callable activation function

			It assumes that norm layer is used before activation.
		*/
		ConvBn2dImpl(const torch::nn::Conv2dOptions &options, BatchNorm::Type norm = BatchNorm::kNone,
			bool activation = false);
		void initialize(const ModelImporter &importer, const std::string &prefix, ModelImporter::Fill fill);

		torch::Tensor forward(torch::Tensor x);

	public:
		torch::nn::Conv2d m_conv{ nullptr };
		BatchNorm m_bn{ nullptr };
		bool m_activation; // relu
	};
	TORCH_MODULE(ConvBn2d);
}