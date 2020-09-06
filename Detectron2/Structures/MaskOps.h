#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from layers/mask_ops.py

	class MaskOps {
	public:
		/**
			Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
			The location, height, and width for pasting each mask is determined by their
			corresponding bounding boxes in boxes.

			Note:
				This is a complicated but more accurate implementation. In actual deployment, it is
				often enough to use a faster but less accurate implementation.
				See :func:`paste_mask_in_image_old` in this file for an alternative implementation.

			Args:
				masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
					detected object instances in the image and Hmask, Wmask are the mask width and mask
					height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
				boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
					boxes[i] and masks[i] correspond to the same object instance.
				image_shape (tuple): height, width
				threshold (float): A threshold in [0, 1] for converting the (soft) masks to
					binary masks.

			Returns:
				img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
				number of detected object instances and Himage, Wimage are the image width
				and height. img_masks[i] is a binary mask for object instance i.
		*/
		static torch::Tensor paste_masks_in_image(torch::Tensor masks, torch::Tensor boxes,
			const ImageSize &image_shape, float threshold = 0.5);


		// The below are the original paste function (from Detectron1) which has
		// larger quantization error.
		// It is faster on CPU, while the aligned one is faster on GPU thanks to grid_sample.

		/**
			Paste a single mask in an image.
			This is a per-box implementation of :func:`paste_masks_in_image`.
			This function has larger quantization error due to incorrect pixel
			modeling and is not used any more.

			Args:
				mask (Tensor): A tensor of shape (Hmask, Wmask) storing the mask of a single
					object instance. Values are in [0, 1].
				box (Tensor): A tensor of shape (4, ) storing the x0, y0, x1, y1 box corners
					of the object instance.
				img_h, img_w (int): Image height and width.
				threshold (float): Mask binarization threshold in [0, 1].

			Returns:
				im_mask (Tensor):
					The resized and binarized object mask pasted into the original
					image plane (a tensor of shape (img_h, img_w)).
		*/
		static torch::Tensor paste_mask_in_image_old(torch::Tensor mask, torch::Tensor box, int img_h, int img_w,
			float threshold);

		// Our pixel modeling requires extrapolation for any continuous
		// coordinate < 0.5 or > length - 0.5. When sampling pixels on the masks,
		// we would like this extrapolation to be an interpolation between boundary values and zero,
		// instead of using absolute zero or boundary values.
		// Therefore `paste_mask_in_image_old` is often used with zero padding around the masks like this:
		// masks, scale = pad_masks(masks[:, 0, :, :], 1)
		// boxes = scale_boxes(boxes.tensor, scale)

		/**
			Args:
				masks (tensor): A tensor of shape (B, M, M) representing B masks.
				padding (int): Number of cells to pad on all sides.

			Returns:
				The padded masks and the scale factor of the padding size / original size.
		*/
		static std::tuple<torch::Tensor, float> pad_masks(torch::Tensor masks, int padding);

		/**
			Args:
				boxes (tensor): A tensor of shape (B, 4) representing B boxes with 4
					coords representing the corners x0, y0, x1, y1,
				scale (float): The box scaling factor.

			Returns:
				Scaled boxes.
		*/
		static torch::Tensor scale_boxes(torch::Tensor boxes, float scale);

	private:
		/**
			Args:
				masks: N, 1, H, W
				boxes: N, 4
				img_h, img_w (int):
				skip_empty (bool): only paste masks within the region that
					tightly bound all boxes, and returns the results this region only.
					An important optimization for CPU.

			Returns:
				if skip_empty == False, a mask of shape (N, img_h, img_w)
				if skip_empty == True, a mask of shape (N, h', w'), and the slice
					object for the corresponding region.
		*/
		static std::tuple<torch::Tensor, TensorVec> _do_paste_mask(torch::Tensor masks, torch::Tensor boxes,
			int img_h, int img_w, bool skip_empty = true);
	};
}