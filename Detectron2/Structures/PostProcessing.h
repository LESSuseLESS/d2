#pragma once

#include "Instances.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/postprocessing.py

	class PostProcessing {
	public:
		/**
			Resize the output instances.
			The input images are often resized when entering an object detector.
			As a result, we often need the outputs of the detector in a different
			resolution from its inputs.

			This function will resize the raw outputs of an R-CNN detector
			to produce outputs according to the desired output resolution.

			Args:
				results (Instances): the raw outputs from the detector.
					`results.image_size` contains the input image resolution the detector sees.
					This object might be modified in-place.
				output_height, output_width: the desired output resolution.

			Returns:
				Instances: the resized output from the model, based on the output resolution
		*/
		static InstancesPtr detector_postprocess(const InstancesPtr &results,
			int output_height, int output_width, float mask_threshold = 0.5);

		/**
			Return semantic segmentation predictions in the original resolution.

			The input images are often resized when entering semantic segmentor. Moreover, in same
			cases, they also padded inside segmentor to be divisible by maximum network stride.
			As a result, we often need the predictions of the segmentor in a different
			resolution from its inputs.

			Args:
				result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
					where C is the number of classes, and H, W are the height and width of the prediction.
				img_size (tuple): image size that segmentor is taking as input.
				output_height, output_width: the desired output resolution.

			Returns:
				semantic segmentation prediction (Tensor): A tensor of the shape
					(C, output_height, output_width) that contains per-pixel soft predictions.
		*/
		static torch::Tensor sem_seg_postprocess(torch::Tensor result, const ImageSize &img_size,
			int output_height, int output_width);
	};
}