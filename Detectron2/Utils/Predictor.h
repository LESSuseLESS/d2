#pragma once

#include <Detectron2/Structures/Instances.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class Predictor {
	public:
		virtual ~Predictor() {}

		/**
		Args:
			original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

		Returns:
			predictions (dict):
				the output of the model for one image only.
				See :doc:`/tutorials/models` for details about the format.
		*/
		virtual InstancesPtr predict(torch::Tensor original_image) = 0;
	};
}
