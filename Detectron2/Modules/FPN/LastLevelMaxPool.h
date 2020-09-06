#pragma once

#include "TopBlock.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/fpn.py

	// This module is used in the original FPN to generate a downsampled P6 feature from P5.
	class LastLevelMaxPoolImpl : public TopBlockImpl {
	public:
		LastLevelMaxPoolImpl();

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;
		virtual TensorVec forward(torch::Tensor x) override;
	};
}