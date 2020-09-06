#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from utils/colormap.py
	// converted from utils/visualizer.py

	// Enum of different color modes to use for instance visualizations.
	enum ColorMode {
		// Picks a random color for every instance and overlay segmentations with low opacity.
		kIMAGE = 0,

		/*
			Let instances of the same category have similar colors
			(from metadata.thing_colors), and overlay them with
			high opacity. This provides more attention on the quality of segmentation.
		*/
		kSEGMENTATION = 1,

		/*
			Same as IMAGE, but convert all areas without masks to gray - scale.
			Only available for drawing per - instance mask predictions.
		*/
		kIMAGE_BW = 2
	};

	using VisColor = std::vector<float>;

	VisColor color_from_tensor(const torch::Tensor &t);
	VisColor color_normalize(const VisColor &color); // each / 255
	VisColor color_denormalize(const VisColor &color); // each * 255

	/**
		Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
		less or more saturation than the original color.

		Args:
			color: color of the polygon. Refer to `matplotlib.colors` for a full list of
				formats that are accepted.
			brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
				0 will correspond to no change, a factor in [-1.0, 0) range will result in
				a darker color and a factor in (0, 1.0] range will result in a lighter color.

		Returns:
			modified_color (tuple[double]): a tuple containing the RGB values of the
				modified color. Each value in the tuple is in the [0.0, 1.0] range.
	*/
	VisColor color_brightness(const VisColor &color, float brightness_factor);
	VisColor color_at_least(const VisColor &color, float minimum); // avoid dark

	/**
		Randomly modifies given color to produce a slightly different color than the color given.

		Args:
			color (tuple[double]): a tuple of 3 elements, containing the RGB values of the color
				picked. The values in the list are in the [0.0, 1.0] range.

		Returns:
			jittered_color (tuple[double]): a tuple of 3 elements, containing the RGB values of the
				color after being jittered. The values in the list are in the [0.0, 1.0] range.
	*/
	VisColor color_jitter(const VisColor &color);

	/**
		rgb(bool) : whether to return RGB colors or BGR colors.
		maximum(int) : either 255 or 1
	*/
	VisColor color_random(bool rgb = true, int maximum = 1);

	struct ClassColor {
		std::string cls;
		VisColor color;
	};
}
