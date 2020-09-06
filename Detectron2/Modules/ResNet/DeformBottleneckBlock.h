#pragma once

#include "CNNBlockBase.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/resnet.py

	// Similar to :class:`BottleneckBlock`, but with :paper:`deformable conv <deformconv>` in the 3x3 convolution.
	class DeformBottleneckBlockImpl : public CNNBlockBaseImpl {
	public:
		DeformBottleneckBlockImpl(int in_channels, int out_channels, int bottleneck_channels, int stride,
			int num_groups, BatchNorm::Type norm, bool stride_in_1x1, int dilation,
			bool deform_modulated, int deform_num_groups);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;
		virtual torch::Tensor forward(torch::Tensor x) override;

	private:
		bool m_deform_modulated;

		ConvBn2d m_shortcut{ nullptr };
		ConvBn2d m_convbn1;
		ConvBn2d m_convbn2_offset;
		ModulePtr m_convbn2{ nullptr };
		ConvBn2d m_convbn3;
	};
}