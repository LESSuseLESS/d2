#pragma once

#include "ROIPoolerLevel.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from layers/roi_align.py

	class ROIAlignImpl : public ROIPoolerLevelImpl {
	public:
		/**
			Args:
				output_size (tuple): h, w
				spatial_scale (float): scale the input boxes by this number
				sampling_ratio (int): number of inputs samples to take for each output
					sample. 0 to take samples densely.
				aligned (bool): if False, use the legacy implementation in
					Detectron. If True, align the results more perfectly.

			Note:
				The meaning of aligned=True:

				Given a continuous coordinate c, its two neighboring pixel indices (in our
				pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
				c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
				from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
				roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
				pixel indices and therefore it uses pixels with a slightly incorrect alignment
				(relative to our pixel model) when performing bilinear interpolation.

				With `aligned=True`,
				we first appropriately scale the ROI and then shift it by -0.5
				prior to calling roi_align. This produces the correct neighbors; see
				detectron2/tests/test_roi_align.py for verification.

				The difference does not make a difference to the model's performance if
				ROIAlign is used together with conv layers.
		*/
		ROIAlignImpl(const Size2D &output_size, float spatial_scale, int sampling_ratio, bool aligned = true);

		// input: NCHW images
		// rois : Bx5 boxes.First column is the index into N.The other 4 columns are xyxy.
		virtual torch::Tensor forward(const torch::Tensor &input, const torch::Tensor &rois) override;
		virtual std::string toString() const override;

	private:
		Size2D m_output_size;
		float m_spatial_scale;
		int m_sampling_ratio;
		bool m_aligned;
	};
}