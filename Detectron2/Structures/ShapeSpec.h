#pragma once

#include "Boxes.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from layers/shape_spec.py

	/**
		A simple structure that contains basic shape specification about a tensor.
		It is often used as the auxiliary inputs/outputs of models,
		to obtain the shape inference ability among pytorch modules.

		Attributes:
			channels:
			height:
			width:
			stride:
	*/
	struct ShapeSpec {
		int channels;
		int height;
		int width;
		int	stride;
		int index;

		using Map = std::unordered_map<std::string, ShapeSpec>;
		using Vec = std::vector<ShapeSpec>;

		int64_t prod() const { return channels * height * width; }
		static Vec filter(const Map &shapes, const std::vector<std::string> &names);
		static std::vector<int> channels_vec(const Vec &shapes);
		static int channels_single(const Vec &shapes);
		static std::vector<int> strides_vec(const Vec &shapes);
	};
}
