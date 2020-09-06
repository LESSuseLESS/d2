#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from structures/image_list.py

	/**
		Structure that holds a list of images (of possibly
		varying sizes) as a single tensor.
		This works by padding the images to the same size,
		and storing in a field the original sizes of each image

		Attributes:
			image_sizes (list[tuple[int, int]]): each tuple is (h, w)
	*/
	class ImageList {
	public:
		/**
			Args:
				tensors: a tuple or list of `torch.Tensors`, each of shape (Hi, Wi) or
					(C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
					to the same shape with `pad_value`.
				size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
					the common height and width is divisible by `size_divisibility`.
					This depends on the model and many models need a divisibility of 32.
				pad_value (float): value to pad

			Returns:
				an `ImageList`.
		*/
		static ImageList from_tensors(const TensorVec &tensors, int size_divisibility = 0, double pad_value = 0.0);

	public:
		/**
			tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
			image_sizes (list[tuple[int, int]]): Each tuple is (h, w). It can
				be smaller than (H, W) due to padding.
		*/
		ImageList(torch::Tensor tensor, std::vector<ImageSize> image_sizes) :
			m_tensor(tensor), m_image_sizes(std::move(image_sizes)) {
		}

		int length() const {
			return m_image_sizes.size();
		}
		const std::vector<ImageSize> &image_sizes() const {
			return m_image_sizes;
		}

		torch::Tensor tensor() const {
			return m_tensor;
		}
		torch::Device device() const {
			return m_tensor.device();
		}

		ImageList to(torch::Device device) {
			std::vector<ImageSize> image_sizes = m_image_sizes;
			return ImageList(m_tensor.to(device), std::move(image_sizes));
		}

		/**
			Access the individual image in its original size.

			Returns:
				Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
		*/
		torch::Tensor get(int64_t idx);

	private:
		torch::Tensor m_tensor;
		std::vector<ImageSize> m_image_sizes;
	};
}
