#pragma once

#include "Transform.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from data/transform/transform.py

	/**
		Resize the image to a target size.
	*/
	class ResizeTransform : public Transform {
	public:
		/**
			h, w (int): original image size
			new_h, new_w (int): new image size
			interp: PIL interpolation methods, defaults to bilinear.
		*/
		ResizeTransform(int h, int w, int new_h, int new_w, Interp interp = kBILINEAR);

		virtual torch::Tensor apply_image(torch::Tensor img, Interp interp = kNone) override;
		virtual torch::Tensor apply_coords(torch::Tensor coords) override;
		virtual torch::Tensor apply_segmentation(torch::Tensor segmentation) override;
		virtual std::shared_ptr<Transform> inverse() override;

	private:
		int m_h;
		int m_w;
		int m_new_h;
		int m_new_w;
		Interp m_interp;
	};
}
