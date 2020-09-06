#pragma once

#include "TopBlock.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/fpn.py

	// This module is used in RetinaNet to generate extra layers, P6 and P7 from C5 feature.
	class LastLevelP6P7Impl : public TopBlockImpl {
	public:
		LastLevelP6P7Impl(int64_t in_channels, int64_t out_channels, const char *in_feature);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;
		virtual TensorVec forward(torch::Tensor c5) override;

	private:
		torch::nn::Conv2d m_p6;
		torch::nn::Conv2d m_p7;
	};
}