#pragma once

#include "Masks.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from structures/masks.py
	
	class PolygonMasks;
	using BitMasksList = TensorVec;

	/**
		This class stores the segmentation masks for all objects in one image, in
		the form of bitmaps.

		Attributes:
			tensor: bool Tensor of N,H,W, representing N instances in the image.
	*/
	class BitMasks : public Masks {
	public:
		/**
			Concatenates a list of BitMasks into a single BitMasks

			Arguments:
				bitmasks_list (list[BitMasks])

			Returns:
				BitMasks: the concatenated BitMasks
		*/
		static BitMasks cat(const BitMasksList &bitmasks_list) {
			return torch::cat(bitmasks_list);
		}

	public:
		// tensor: bool Tensor of N,H,W, representing N instances in the image.
		BitMasks(const torch::Tensor &tensor);
		BitMasks(const BitMasks &bitmasks);

		// implementing Sequence
		virtual int size() const override { return m_tensor.size(0); }
		virtual std::string toString() const override;
		virtual SequencePtr slice(int64_t start, int64_t end) const override;
		virtual SequencePtr index(torch::Tensor item) const override;
		virtual SequencePtr cat(const std::vector<SequencePtr> &seqs, int total) const override;

		const torch::Tensor &tensor() const { return m_tensor; }
		BitMasks to(torch::Device device) {
			return BitMasks(m_tensor.to(device));
		}
		torch::Device device() {
			return m_tensor.device();
		}

		/**
			Returns:
				BitMasks: Create a new :class:`BitMasks` by indexing.

			The following usage are allowed:

			1. `new_masks = masks[3]`: return a `BitMasks` which contains only one mask.
			2. `new_masks = masks[2:10]`: return a slice of masks.
			3. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
			   with `length = len(masks)`. Nonzero elements in the vector will be selected.

			Note that the returned object might share storage with this object,
			subject to Pytorch's indexing semantics.
		*/
		BitMasks operator[](int64_t item) const;
		BitMasks operator[](torch::ArrayRef<torch::indexing::TensorIndex> item) const;

		/**
			Find masks that are non-empty.

			Returns:
				Tensor: a BoolTensor which represents
					whether each mask is empty (False) or non-empty (True).
		*/
		torch::Tensor nonempty() {
			return m_tensor.flatten(1).any(1);
		}

		/**
			Crop each bitmask by the given box, and resize results to (mask_size, mask_size).
			This can be used to prepare training targets for Mask R-CNN.
			It has less reconstruction error compared to rasterization with polygons.
			However we observe no difference in accuracy,
			but BitMasks requires more memory to store all the masks.

			Args:
				boxes (Tensor): Nx4 tensor storing the boxes for each mask
				mask_size (int): the size of the rasterized mask.

			Returns:
				Tensor:
					A bool tensor of shape (N, mask_size, mask_size), where
					N is the number of predicted boxes for this image.
		*/
		virtual torch::Tensor crop_and_resize(torch::Tensor boxes, int mask_size) override;

		void get_bounding_boxes() {
			assert(false);
		}

	private:
		torch::Tensor m_tensor;
		ImageSize m_image_size;
	};
}