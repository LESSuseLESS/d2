#pragma once

#include <Detectron2/Utils/Canvas.h>
#include <Detectron2/Structures/PanopticSegment.h>
#include <Detectron2/Structures/GenericMask.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from utils/visualizer.py

	class _PanopticPrediction {
	public:
		_PanopticPrediction(const torch::Tensor &panoptic_seg, const std::vector<SegmentInfo> &segments_info);

		/**
			Returns:
				(H, W) array, a mask for all pixels that have a prediction
		*/
		torch::Tensor non_empty_mask() const;

		void semantic_masks(std::function<void(torch::Tensor, const SegmentInfo &)> func) const;
		void instance_masks(std::function<void(torch::Tensor, const SegmentInfo &)> func) const;

	private:
		torch::Tensor m_seg;
		std::map<int, SegmentInfo> m_sinfo; // seg id -> seg info
		torch::Tensor m_seg_ids;
		torch::Tensor m_seg_areas;
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class VisImage {
	public:
		VisImage() {}
		VisImage(const VisImage &) = default;
		VisImage &operator=(const VisImage &) = default;

		/**
			img (ndarray): an RGB image of shape (H, W, 3).
			scale (float): scale the input image
		*/
		VisImage(const torch::Tensor &img, float scale = 1.0);

		torch::Tensor &img() { return m_img; }
		const torch::Tensor &img() const { return m_img; }
		int height() const { return m_height; }
		int width() const { return m_width; }
		float scale() const { return m_scale; }

		// so we can draw into this object
		std::shared_ptr<Canvas> get_canvas() {
			return m_canvas;
		}

		/**
		Args:
			filepath (str): a string that contains the absolute path, including the file name, where
				the visualized image will be saved.
		*/
		void save(const std::string &filepath) const;

		/**
		Returns:
			ndarray:
				the visualized image of shape (H, W, 3) (RGB) in uint8 type.
				The shape is scaled w.r.t the input image using the given `scale` argument.
		*/
		torch::Tensor get_image() const;

	private:
		torch::Tensor m_img; // original image
		float m_scale;
		int m_height;
		int m_width;
		std::shared_ptr<Canvas> m_canvas; // patches we drew

		/**
			Args:
				Same as in :meth:`__init__()`.

			Returns:
				fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
				ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
		*/
		void _setup_figure();
	};
}
