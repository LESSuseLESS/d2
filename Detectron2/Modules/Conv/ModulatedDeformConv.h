#pragma once

#include <Detectron2/Modules/BatchNorm/BatchNorm.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from layers/deform_conv.py

	class ModulatedDeformConvImpl : public torch::nn::Module {
	public:
		/**
			Modulated deformable convolution from :paper:`deformconv2`.

			Arguments are similar to :class:`Conv2D`. Extra arguments:

			Args:
				deformable_groups (int): number of groups used in deformable convolution.
				norm (nn.Module, optional): a normalization layer
				activation (callable(Tensor) -> Tensor): a callable activation function
		*/
		ModulatedDeformConvImpl(int in_channels, int out_channels, int kernel_size, int stride, int padding,
			int dilation, int groups, int deformable_groups, bool bias, BatchNorm::Type norm, bool activation = false);
		void initialize(const ModelImporter &importer, const std::string &prefix, ModelImporter::Fill fill);

		torch::Tensor forward(torch::Tensor x, torch::Tensor offset, torch::Tensor mask);

		std::string extra_repr() const;

	public:
		int m_in_channels;
		int m_out_channels;
		std::vector<int> m_kernel_size;
		int m_stride;
		int m_padding;
		int m_dilation;
		int m_groups;
		int m_deformable_groups;
		bool m_with_bias;
		BatchNorm m_bn;
		bool m_activation; // relu

		torch::Tensor m_weight;
		torch::Tensor m_bias;
	};
	TORCH_MODULE(ModulatedDeformConv);
}