#pragma once

#include "PolygonMasks.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from utils/visualizer.py

	/**
		Attribute:
			polygons (list[ndarray]): list[ndarray]: polygons for this mask.
				Each ndarray has format [x, y, x, y, ...]
			mask (ndarray): a binary mask
	*/
	class GenericMask {
	public:
		static torch::Tensor toCocoMask(const std::vector<std::shared_ptr<GenericMask>> &masks);

		static std::tuple<TensorVec, int> mask_to_polygons(const torch::Tensor &mask);

		static std::vector<std::shared_ptr<GenericMask>>
			_convert_masks(const BitMasks &m, int height, int width);
		static std::vector<std::shared_ptr<GenericMask>>
			_convert_masks(const PolygonMasks &m, int height, int width);

	public:
		GenericMask(const torch::Tensor &mask, int height, int width);
		GenericMask(const TensorVec &polygons, int height, int width);
		GenericMask(const mask_util::MaskObject &obj, int height, int width);

		torch::Tensor polygons_to_mask(const TensorVec &polygons);

		torch::Tensor mask();
		TensorVec polygons();
		bool has_holes();

		float area() const {
			return m_mask.sum().item<float>();
		}

		torch::Tensor bbox() const;

	private:
		int m_height;
		int m_width;
		torch::Tensor m_mask;
		TensorVec m_polygons;
		bool m_has_mask;
		bool m_has_polygons;
		int m_has_holes;
	};
}
