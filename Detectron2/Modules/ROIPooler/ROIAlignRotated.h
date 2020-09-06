#pragma once

#include "ROIPoolerLevel.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from layers/roi_align_rotated.py

	class ROIAlignRotatedImpl : public ROIPoolerLevelImpl {
	public:
		/**
			Args:
				output_size (tuple): h, w
				spatial_scale (float): scale the input boxes by this number
				sampling_ratio (int): number of inputs samples to take for each output
					sample. 0 to take samples densely.

			Note:
				ROIAlignRotated supports continuous coordinate by default:
				Given a continuous coordinate c, its two neighboring pixel indices (in our
				pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
				c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
				from the underlying signal at continuous coordinates 0.5 and 1.5).
		*/
		ROIAlignRotatedImpl(const Size2D &output_size, float spatial_scale, int sampling_ratio);

		// input: NCHW images
		// rois : Bx6 boxes.First column is the index into N. The other 5 columns are
		//		(x_ctr, y_ctr, width, height, angle_degrees).
		virtual torch::Tensor forward(const torch::Tensor &input, const torch::Tensor &rois) override;
		virtual std::string toString() const override;

	private:
		Size2D m_output_size;
		float m_spatial_scale;
		int m_sampling_ratio;
	};
}