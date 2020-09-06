#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/fpn.py

	class TopBlockImpl : public torch::nn::Module {
	public:
		virtual ~TopBlockImpl() {}
		virtual void initialize(const ModelImporter &importer, const std::string &prefix) = 0;
		virtual TensorVec forward(torch::Tensor x) = 0;

		int num_levels() const { return m_num_levels; }
		std::string in_feature() const { return m_in_feature; }

	protected:
		int m_num_levels;			// the number of extra FPN levels added by this block
		std::string m_in_feature;	// a string representing its input feature (e.g., p5).
	};
	TORCH_MODULE(TopBlock);
}