#pragma once

#include "ROIPoolerLevel.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/poolers.py

	// Region of interest feature map pooler that supports pooling from one or more feature maps.
	class ROIPoolerImpl : public torch::nn::Module {
	public:
		/*
			Map each box in `box_lists` to a feature map level index and return the assignment vector.

			Args:
				box_lists (list[Boxes] | list[RotatedBoxes]): A list of N Boxes or N RotatedBoxes,
					where N is the number of images in the batch.
				min_level (int): Smallest feature map level index. The input is considered index 0,
					the output of stage 1 is index 1, and so.
				max_level (int): Largest feature map level index.
				canonical_box_size (int): A canonical box size in pixels (sqrt(box area)).
				canonical_level (int): The feature map level index on which a canonically-sized box
					should be placed.

			Returns:
				A tensor of length M, where M is the total number of boxes aggregated over all
					N batch images. The memory layout corresponds to the concatenation of boxes
					from all images. Each element is the feature map index, as an offset from
					`self.min_level`, for the corresponding box (so value i means the box is at
					`self.min_level + i`).
		*/
		static torch::Tensor assign_boxes_to_levels(const BoxesList &box_lists, int min_level, int max_level,
			int canonical_box_size, int canonical_level);

		/**
			Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
			(see description under Returns).

			Args:
				box_lists (list[Boxes] | list[RotatedBoxes]):
					A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.

			Returns:
				When input is list[Boxes]:
					A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
					N batch images.
					The 5 columns are (batch index, x0, y0, x1, y1), where batch index
					is the index in [0, N) identifying which batch image the box with corners at
					(x0, y0, x1, y1) comes from.
				When input is list[RotatedBoxes]:
					A tensor of shape (M, 6), where M is the total number of boxes aggregated over all
					N batch images.
					The 6 columns are (batch index, x_ctr, y_ctr, width, height, angle_degrees),
					where batch index is the index in [0, N) identifying which batch image the
					rotated box (x_ctr, y_ctr, width, height, angle_degrees) comes from.
		*/
		static torch::Tensor convert_boxes_to_pooler_format(const BoxesList &box_lists);

	public:
		/**
			output_size: output size of the pooled region, e.g., 14 x 14.
			scales: The scale for each low-level pooling op relative to
				the input image. For a feature map with stride s relative to the input
				image, scale is defined as a 1 / s. The stride must be power of 2.
				When there are multiple scales, they must form a pyramid, i.e. they must be
				a monotically decreasing geometric sequence with a factor of 1/2.
			sampling_ratio: The `sampling_ratio` parameter for the ROIAlign op.
			pooler_type: Name of the type of pooling operation that should be applied.
				For instance, "ROIPool" or "ROIAlignV2".
			canonical_box_size: A canonical box size in pixels (sqrt(box area)). The default
				is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
				pre-training).
			canonical_level: The feature map level index from which a canonically-sized box
				should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
				i.e., a box of size 224x224 will be placed on the feature with stride=16.
				The box placement for all boxes will be determined from their sizes w.r.t
				canonical_box_size. For example, a box whose area is 4x that of a canonical box
				should be used to pool features from feature level ``canonical_level+1``.

				Note that the actual input feature maps given to this module may not have
				sufficiently many levels for the input boxes. If the boxes are too large or too
				small for the input feature maps, the closest level will be used.
		 */
		ROIPoolerImpl(const std::string &pooler_type, const Size2D &output_size,
			const std::vector<float> &scales, int sampling_ratio, int canonical_box_size = 224,
			int canonical_level = 4);

		/*
			Args:
				x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
					used to construct this module.
				box_lists (list[Boxes] | list[RotatedBoxes]):
					A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
					The box coordinates are defined on the original image and
					will be scaled by the `scales` argument of :class:`ROIPooler`.

			Returns:
				Tensor:
					A tensor of shape (M, C, output_size, output_size) where M is the total number of
					boxes aggregated over all N batch images and C is the number of channels in `x`.
		*/
		torch::Tensor forward(const TensorVec &x, const BoxesList &box_lists);

	private:
		Size2D m_output_size;
		int m_canonical_box_size;
		int m_canonical_level;

		std::vector<ROIPoolerLevel> m_level_poolers;
		int m_min_level;
		int m_max_level;
	};
	TORCH_MODULE(ROIPooler);
}