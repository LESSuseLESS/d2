#pragma once

#include "Transform.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from data/transform/transform_gen.py

    /**
		TransformGen takes an image of type uint8 in range [0, 255], or
		floating point in range [0, 1] or [0, 255] as input.

		It creates a :class:`Transform` based on the given image, sometimes with randomness.
		The transform can then be used to transform images
		or other data (boxes, points, annotations, etc.) associated with it.

		The assumption made in this class
		is that the image itself is sufficient to instantiate a transform.
		When this assumption is not true, you need to create the transforms by your own.

		A list of `TransformGen` can be applied with :func:`apply_transform_gens`.
	*/
	class TransformGen {
	public:
		TransformGen(const std::unordered_map<std::string, YAML::Node> &params = {});
		virtual ~TransformGen() {}

		virtual std::shared_ptr<Transform> get_transform(torch::Tensor img) = 0;

		/**
			Produce something like:
			"MyTransformGen(field1={self.field1}, field2={self.field2})"
		*/
		std::string repr() const;
		std::string str() const {
			return repr();
		}

	protected:
		std::unordered_map<std::string, YAML::Node> m_attrs;

		// Uniform float random number between low and high.
		static torch::Tensor _rand_range(double low = 1.0, double *high = nullptr, torch::IntArrayRef size = {});
	};
}
