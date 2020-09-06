#pragma once

#include "ROIPoolerLevel.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from https://github.com/pytorch/vision torchvision/ops/roi_pool.py

    /**
		Performs Region of Interest (RoI) Pool operator described in Fast R-CNN

		Arguments:
			input (Tensor[N, C, H, W]): input tensor
			boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
				format where the regions will be taken from. If a single Tensor is passed,
				then the first column should contain the batch index. If a list of Tensors
				is passed, then each Tensor will correspond to the boxes for an element i
				in a batch
			output_size (int or Tuple[int, int]): the size of the output after the cropping
				is performed, as (height, width)
			spatial_scale (float): a scaling factor that maps the input coordinates to
				the box coordinates. Default: 1.0

		Returns:
			output (Tensor[K, C, output_size[0], output_size[1]])
	*/
	torch::Tensor roi_pool(const torch::Tensor &input, const torch::Tensor &boxes, const Size2D &output_size,
		float spatial_scale = 1.0);
	torch::Tensor roi_pool(const torch::Tensor &input, const BoxesList &boxes, const Size2D &output_size,
		float spatial_scale = 1.0);

	class RoIPoolImpl : public ROIPoolerLevelImpl {
	public:
		RoIPoolImpl(const Size2D &output_size, float spatial_scale);

		virtual torch::Tensor forward(const torch::Tensor &input, const torch::Tensor &rois) override;
		virtual std::string toString() const override;

	private:
		Size2D m_output_size;
		float m_spatial_scale;
	};
}