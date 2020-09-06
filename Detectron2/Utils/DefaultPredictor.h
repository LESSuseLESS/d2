#pragma once

#include "Predictor.h"
#include "VisImage.h"
#include <Detectron2/Data/MetadataCatalog.h>
#include <Detectron2/Data/TransformGen.h>
#include <Detectron2/MetaArch/MetaArch.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from engine/defaults.py

    /**
		Create a simple end-to-end predictor with the given config that runs on
		single device for a single input image.

		Compared to using the model directly, this class does the following additions:

		1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
		2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
		3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
		4. Take one input image and produce a single output, instead of a batch.

		If you'd like to do anything more fancy, please refer to its source code
		as examples to build and use the model manually.

		Attributes:
			metadata (Metadata): the metadata of the underlying dataset, obtained from
				cfg.DATASETS.TEST.

		Examples:

		.. code-block:: python

			pred = DefaultPredictor(cfg)
			inputs = cv2.imread("input.jpg")
			outputs = pred(inputs)
	*/
	class DefaultPredictor : public Predictor {
	public:
		DefaultPredictor(const CfgNode &cfg);

		/**
		Args:
			original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

		Returns:
			predictions (dict):
				the output of the model for one image only.
				See :doc:`/tutorials/models` for details about the format.
		*/
		virtual InstancesPtr predict(torch::Tensor original_image) override;

	protected:
		CfgNode m_cfg;
		MetaArch m_model;
		Metadata m_metadata;
		std::shared_ptr<TransformGen> m_transform_gen;
		std::string m_input_format;
	};
}
