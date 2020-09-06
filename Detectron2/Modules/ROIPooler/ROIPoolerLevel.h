#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class ROIPoolerLevelImpl : public torch::nn::Module {
	public:
		virtual ~ROIPoolerLevelImpl() {}
		virtual torch::Tensor forward(const torch::Tensor &input, const torch::Tensor &rois) = 0;
		virtual std::string toString() const = 0;
	};
	TORCH_MODULE(ROIPoolerLevel);
}