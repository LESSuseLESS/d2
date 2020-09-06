#pragma once

#include "BitMasks.h"

#include <Detectron2/coco/mask.h>
#include <Detectron2/Structures/Boxes.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from structures/masks.py

	class PolygonMasks;
	using PolygonMasksList = std::vector<std::shared_ptr<PolygonMasks>>;
	namespace mask_util = pycocotools;

    /**
		This class stores the segmentation masks for all objects in one image, in the form of polygons.

		Attributes:
			polygons: list[list[ndarray]]. Each ndarray is a float64 vector representing a polygon.
	*/
	class PolygonMasks : public Masks {
	public:
		/**
			Concatenates a list of PolygonMasks into a single PolygonMasks

			Arguments:
				polymasks_list (list[PolygonMasks])

			Returns:
				PolygonMasks: the concatenated PolygonMasks
		*/
		static PolygonMasks cat(const PolygonMasksList &polymasks_list);

		static float polygon_area(const torch::Tensor &x, const torch::Tensor &y);

		/**
			Args:
				polygons (list[ndarray]): each array has shape (Nx2,)
				height, width (int)

			Returns:
				ndarray: a bool mask of shape (height, width)
		*/
		static torch::Tensor polygons_to_bitmask(const TensorVec &polygons, int height, int width);

		/**
			Rasterize the polygons into a mask image and
			crop the mask content in the given box.
			The cropped mask is resized to (mask_size, mask_size).

			This function is used when generating training targets for mask head in Mask R-CNN.
			Given original ground-truth masks for an image, new ground-truth mask
			training targets in the size of `mask_size x mask_size`
			must be provided for each predicted box. This function will be called to
			produce such targets.

			Args:
				polygons (list[ndarray[float]]): a list of polygons, which represents an instance.
				box: 4-element numpy array
				mask_size (int):

			Returns:
				Tensor: BoolTensor of shape (mask_size, mask_size)
		*/
		static torch::Tensor rasterize_polygons_within_box(const TensorVec &polygons, const torch::Tensor &box,
			int mask_size);

	public:
		/**
			polygons (list[list[np.ndarray]]): The first
				level of the list correspond to individual instances,
				the second level to all the polygons that compose the
				instance, and the third level to the polygon coordinates.
				The third level array should have the format of
				[x0, y0, x1, y1, ..., xn, yn] (n >= 3).
		*/
		PolygonMasks(std::vector<TensorVec> polygons);

		// implementing Sequence
		virtual int size() const override { return m_polygons.size(); }
		virtual std::string toString() const override;
		virtual SequencePtr slice(int64_t start, int64_t end) const override;
		virtual SequencePtr index(torch::Tensor item) const override;
		virtual SequencePtr cat(const std::vector<SequencePtr> &seqs, int total) const override;

		torch::Device device() const {
			return torch::kCPU;
		}

		const std::vector<TensorVec> &polygons() const {
			return m_polygons;
		}

		/**
			polygon_masks (list[list[ndarray]] or PolygonMasks)
			height, width (int)
		*/
		BitMasks from_polygon_masks(int height, int width) const;

		// Returns: Boxes: tight bounding boxes around polygon masks.
		Boxes get_bounding_boxes() const;

		/**
			Find masks that are non-empty.

			Returns:
				Tensor:
					a BoolTensor which represents whether each mask is empty (False) or not (True).
		*/
		torch::Tensor nonempty();

		/**
			Support indexing over the instances and return a `PolygonMasks` object.
			`item` can be:

			1. An integer. It will return an object with only one instance.
			2. A slice. It will return an object with the selected instances.
			3. A list[int]. It will return an object with the selected instances,
			   correpsonding to the indices in the list.
			4. A vector mask of type BoolTensor, whose length is num_instances.
			   It will return an object with the instances whose mask is nonzero.
		*/
		PolygonMasks operator[](int64_t item) const;
		PolygonMasks operator[](const std::vector<int64_t> &item) const;
		//~! PolygonMasks operator[](torch::ArrayRef<torch::indexing::TensorIndex> item) const;

		/**
			Crop each mask by the given box, and resize results to (mask_size, mask_size).
			This can be used to prepare training targets for Mask R-CNN.

			Args:
				boxes (Tensor): Nx4 tensor storing the boxes for each mask
				mask_size (int): the size of the rasterized mask.

			Returns:
				Tensor: A bool tensor of shape (N, mask_size, mask_size), where
				N is the number of predicted boxes for this image.
		*/
		virtual torch::Tensor crop_and_resize(torch::Tensor boxes, int mask_size) override;

		/**
			Computes area of the mask.
			Only works with Polygons, using the shoelace formula:
			https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

			Returns:
				Tensor: a vector, area for each instance
		*/
		torch::Tensor area();

	private:
		std::vector<TensorVec> m_polygons;

		PolygonMasks() {}
	};
}