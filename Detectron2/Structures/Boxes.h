#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from structures/boxes.py

	using BoxSizeType = Size2D;

	/**
		This structure stores a list of boxes as a Nx4 torch.Tensor. It supports some common methods about boxes
		(`area`, `clip`, `nonempty`, etc), and also behaves like a Tensor (support indexing, `to(device)`, `.device`,
		and iteration over all boxes)

		Attributes:
			tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
	*/
	class Boxes {
	public:
		// Concatenates a list of Boxes into a single Boxes
		static torch::Tensor cat(const BoxesList &boxes_list);

		// Creating Boxes or RotatedBoxes, depending on size(-1) == 4 or 5
		static std::shared_ptr<Boxes> boxes(const torch::Tensor &tensor);

		// implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
		// with slight modifications
		/**
			Given two lists of boxes of size N and M,
			compute the IoU (intersection over union)
			between __all__ N x M pairs of boxes.
			The box order must be (xmin, ymin, xmax, ymax).

			Args:
				boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

			Returns:
				Tensor: IoU, sized [N,M].
		*/
		static torch::Tensor pairwise_iou(const Boxes &boxes1, const Boxes &boxes2);

		/**
			Compute pairwise intersection over union (IOU) of two sets of matched
			boxes. The box order must be (xmin, ymin, xmax, ymax).
			Similar to boxlist_iou, but computes only diagonal elements of the matrix
			Arguments:
				boxes1: (Boxes) bounding boxes, sized [N,4].
				boxes2: (Boxes) bounding boxes, sized [N,4].
			Returns:
				(tensor) iou, sized [N].
		*/
		static torch::Tensor matched_boxlist_iou(const Boxes &boxes1, const Boxes &boxes2);

	public:
		// tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
		Boxes(const torch::Tensor &tensor);
		Boxes(const Boxes &boxes);
		virtual ~Boxes() {}

		const torch::Tensor &tensor() const { return m_tensor; }
		Boxes clone() const { return Boxes(m_tensor.clone()); }
		Boxes to(torch::Device device) const { return Boxes(m_tensor.to(device)); }

		int len() const { return m_tensor.size(0); }
		virtual std::string toString() const;
		torch::Device device() const { return m_tensor.device(); }
		std::tuple<int, int, int, int> bbox(int index) const;

		torch::Tensor lefts() const		{ return m_tensor.index({ Colon, 0 }); }
		torch::Tensor tops() const		{ return m_tensor.index({ Colon, 1 }); }
		torch::Tensor rights() const	{ return m_tensor.index({ Colon, 2 }); }
		torch::Tensor bottoms() const	{ return m_tensor.index({ Colon, 3 }); }
		torch::Tensor widths() const	{ return rights() - lefts(); }
		torch::Tensor heights() const	{ return bottoms() - tops(); }

		virtual torch::Tensor area() const { return widths() * heights(); }

		// Clip (in place) the boxes by limiting x coordinates to the range [0, width] and y coordinates to the
		// range[0, height].
		// box_size(height, width) : The clipping box's size.
		virtual void clip(BoxSizeType box_size, float clip_angle_threshold = 1.0);

		// Find boxes that are non-empty. A box is considered empty, if either of its side is no larger than threshold.
		// Returns: a binary vector which represents whether each box is empty (False) or non-empty (True).
		virtual torch::Tensor nonempty(float threshold = 0.0) const;

		/**
			1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
			2. `new_boxes = boxes[2:10]`: return a slice of boxes.
			3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
				with `length = len(boxes)`. Nonzero elements in the vector will be selected.
		*/
		Boxes operator[](int64_t item) const;
		Boxes operator[](torch::ArrayRef<torch::indexing::TensorIndex> item) const;

		// box_size: Size of the reference box.
		// boundary_threshold: Boxes that extend beyond the reference box boundary by more than 
		//		boundary_threshold are considered "outside".
		// Returns: a binary vector, indicating whether each box is inside the reference box.
		virtual torch::Tensor inside_box(BoxSizeType box_size, int boundary_threshold = 0) const;

		// The box centers in a Nx2 array of (x, y).
		virtual torch::Tensor get_centers() const;

		// Scale the box with horizontal and vertical scaling factors
		virtual void scale(float scale_x, float scale_y);

	protected:
		Boxes() {}
		torch::Tensor m_tensor;
	};
}
