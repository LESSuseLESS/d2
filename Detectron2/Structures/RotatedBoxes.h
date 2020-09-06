#pragma once

#include "Boxes.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from structures/rotated_boxes.py
	// converted from layers/rotated_boxes.py

    /**
		This structure stores a list of rotated boxes as a Nx5 torch.Tensor.
		It supports some common methods about boxes
		(`area`, `clip`, `nonempty`, etc),
		and also behaves like a Tensor
		(support indexing, `to(device)`, `.device`, and iteration over all boxes)
	*/
	class RotatedBoxes : public Boxes {
	public:
		/**
			Given two lists of rotated boxes of size N and M,
			compute the IoU (intersection over union)
			between __all__ N x M pairs of boxes.
			The box order must be (x_center, y_center, width, height, angle).

			Args:
				boxes1, boxes2 (RotatedBoxes):
					two `RotatedBoxes`. Contains N & M rotated boxes, respectively.

			Returns:
				Tensor: IoU, sized [N,M].
		*/
		static torch::Tensor pairwise_iou_rotated(const RotatedBoxes &boxes1, const RotatedBoxes &boxes2);

		/**
			Return intersection - over - union (Jaccard index) of boxes.

			Both sets of boxes are expected to be in
			(x_center, y_center, width, height, angle) format.

			Arguments :
			boxes1(Tensor[N, 5])
			boxes2(Tensor[M, 5])

			Returns :
			iou(Tensor[N, M]) : the NxM matrix containing the pairwise
			IoU values for every element in boxes1 and boxes2
		*/
		static torch::Tensor pairwise_iou_rotated(const torch::Tensor &boxes1, const torch::Tensor &boxes2);

	public:
		/**
			Args:
				tensor (Tensor[float]): a Nx5 matrix.  Each row is
					(x_center, y_center, width, height, angle),
					in which angle is represented in degrees.
					While there's no strict range restriction for it,
					the recommended principal range is between [-180, 180) degrees.

			Assume we have a horizontal box B = (x_center, y_center, width, height),
			where width is along the x-axis and height is along the y-axis.
			The rotated box B_rot (x_center, y_center, width, height, angle)
			can be seen as:

			1. When angle == 0:
			   B_rot == B
			2. When angle > 0:
			   B_rot is obtained by rotating B w.r.t its center by :math:`|angle|` degrees CCW;
			3. When angle < 0:
			   B_rot is obtained by rotating B w.r.t its center by :math:`|angle|` degrees CW.

			Mathematically, since the right-handed coordinate system for image space
			is (y, x), where y is top->down and x is left->right, the 4 vertices of the
			rotated rectangle :math:`(yr_i, xr_i)` (i = 1, 2, 3, 4) can be obtained from
			the vertices of the horizontal rectangle (y_i, x_i) (i = 1, 2, 3, 4)
			in the following way (:math:`\\theta = angle*\\pi/180` is the angle in radians,
			(y_c, x_c) is the center of the rectangle):

			.. math::

				yr_i = \\cos(\\theta) (y_i - y_c) - \\sin(\\theta) (x_i - x_c) + y_c,

				xr_i = \\sin(\\theta) (y_i - y_c) + \\cos(\\theta) (x_i - x_c) + x_c,

			which is the standard rigid-body rotation transformation.

			Intuitively, the angle is
			(1) the rotation angle from y-axis in image space
			to the height vector (top->down in the box's local coordinate system)
			of the box in CCW, and
			(2) the rotation angle from x-axis in image space
			to the width vector (left->right in the box's local coordinate system)
			of the box in CCW.

			More intuitively, consider the following horizontal box ABCD represented
			in (x1, y1, x2, y2): (3, 2, 7, 4),
			covering the [3, 7] x [2, 4] region of the continuous coordinate system
			which looks like this:

			.. code:: none

				O--------> x
				|
				|  A---B
				|  |   |
				|  D---C
				|
				v y

			Note that each capital letter represents one 0-dimensional geometric point
			instead of a 'square pixel' here.

			In the example above, using (x, y) to represent a point we have:

			.. math::

				O = (0, 0), A = (3, 2), B = (7, 2), C = (7, 4), D = (3, 4)

			We name vector AB = vector DC as the width vector in box's local coordinate system, and
			vector AD = vector BC as the height vector in box's local coordinate system. Initially,
			when angle = 0 degree, they're aligned with the positive directions of x-axis and y-axis
			in the image space, respectively.

			For better illustration, we denote the center of the box as E,

			.. code:: none

				O--------> x
				|
				|  A---B
				|  | E |
				|  D---C
				|
				v y

			where the center E = ((3+7)/2, (2+4)/2) = (5, 3).

			Also,

			.. math::

				width = |AB| = |CD| = 7 - 3 = 4,
				height = |AD| = |BC| = 4 - 2 = 2.

			Therefore, the corresponding representation for the same shape in rotated box in
			(x_center, y_center, width, height, angle) format is:

			(5, 3, 4, 2, 0),

			Now, let's consider (5, 3, 4, 2, 90), which is rotated by 90 degrees
			CCW (counter-clockwise) by definition. It looks like this:

			.. code:: none

				O--------> x
				|   B-C
				|   | |
				|   |E|
				|   | |
				|   A-D
				v y

			The center E is still located at the same point (5, 3), while the vertices
			ABCD are rotated by 90 degrees CCW with regard to E:
			A = (4, 5), B = (4, 1), C = (6, 1), D = (6, 5)

			Here, 90 degrees can be seen as the CCW angle to rotate from y-axis to
			vector AD or vector BC (the top->down height vector in box's local coordinate system),
			or the CCW angle to rotate from x-axis to vector AB or vector DC (the left->right
			width vector in box's local coordinate system).

			.. math::

				width = |AB| = |CD| = 5 - 1 = 4,
				height = |AD| = |BC| = 6 - 4 = 2.

			Next, how about (5, 3, 4, 2, -90), which is rotated by 90 degrees CW (clockwise)
			by definition? It looks like this:

			.. code:: none

				O--------> x
				|   D-A
				|   | |
				|   |E|
				|   | |
				|   C-B
				v y

			The center E is still located at the same point (5, 3), while the vertices
			ABCD are rotated by 90 degrees CW with regard to E:
			A = (6, 1), B = (6, 5), C = (4, 5), D = (4, 1)

			.. math::

				width = |AB| = |CD| = 5 - 1 = 4,
				height = |AD| = |BC| = 6 - 4 = 2.

			This covers exactly the same region as (5, 3, 4, 2, 90) does, and their IoU
			will be 1. However, these two will generate different RoI Pooling results and
			should not be treated as an identical box.

			On the other hand, it's easy to see that (X, Y, W, H, A) is identical to
			(X, Y, W, H, A+360N), for any integer N. For example (5, 3, 4, 2, 270) would be
			identical to (5, 3, 4, 2, -90), because rotating the shape 270 degrees CCW is
			equivalent to rotating the same shape 90 degrees CW.

			We could rotate further to get (5, 3, 4, 2, 180), or (5, 3, 4, 2, -180):

			.. code:: none

				O--------> x
				|
				|  C---D
				|  | E |
				|  B---A
				|
				v y

			.. math::

				A = (7, 4), B = (3, 4), C = (3, 2), D = (7, 2),

				width = |AB| = |CD| = 7 - 3 = 4,
				height = |AD| = |BC| = 4 - 2 = 2.

			Finally, this is a very inaccurate (heavily quantized) illustration of
			how (5, 3, 4, 2, 60) looks like in case anyone wonders:

			.. code:: none

				O--------> x
				|     B\
				|    /  C
				|   /E /
				|  A  /
				|   `D
				v y

			It's still a rectangle with center of (5, 3), width of 4 and height of 2,
			but its angle (and thus orientation) is somewhere between
			(5, 3, 4, 2, 0) and (5, 3, 4, 2, 90).
		*/
		RotatedBoxes(const torch::Tensor &tensor);
		RotatedBoxes(const RotatedBoxes &boxes);

		RotatedBoxes clone() const { return RotatedBoxes(m_tensor.clone()); }
		RotatedBoxes to(torch::Device device) const { return RotatedBoxes(m_tensor.to(device)); }
		virtual std::string toString() const override;

		// Computes the area of all the boxes.
		// Returns: torch.Tensor: a vector with areas of each box.
		virtual torch::Tensor area() const override;

		// Restrict angles to the range of [-180, 180) degrees
		void normalize_angles();

		/**
			Clip (in place) the boxes by limiting x coordinates to the range [0, width]
			and y coordinates to the range [0, height].

			For RRPN:
			Only clip boxes that are almost horizontal with a tolerance of
			clip_angle_threshold to maintain backward compatibility.

			Rotated boxes beyond this threshold are not clipped for two reasons:

			1. There are potentially multiple ways to clip a rotated box to make it
			   fit within the image.
			2. It's tricky to make the entire rectangular box fit within the image
			   and still be able to not leave out pixels of interest.

			Therefore we rely on ops like RoIAlignRotated to safely handle this.

			Args:
				box_size (height, width): The clipping box's size.
				clip_angle_threshold:
					Iff. abs(normalized(angle)) <= clip_angle_threshold (in degrees),
					we do the clipping as horizontal boxes.
		*/
		virtual void clip(BoxSizeType box_size, float clip_angle_threshold = 1.0) override;


		/**
			Find boxes that are non-empty.
			A box is considered empty, if either of its side is no larger than threshold.

			Returns:
				Tensor: a binary vector which represents
				whether each box is empty (False) or non-empty (True).
		*/
		virtual torch::Tensor nonempty(float threshold = 0.0) const override;

		/**
			Returns:
				RotatedBoxes: Create a new :class:`RotatedBoxes` by indexing.

			The following usage are allowed:

			1. `new_boxes = boxes[3]`: return a `RotatedBoxes` which contains only one box.
			2. `new_boxes = boxes[2:10]`: return a slice of boxes.
			3. `new_boxes = boxes[vector]`, where vector is a torch.ByteTensor
			   with `length = len(boxes)`. Nonzero elements in the vector will be selected.

			Note that the returned RotatedBoxes might share storage with this RotatedBoxes,
			subject to Pytorch's indexing semantics.
		*/
		RotatedBoxes operator[](int64_t item) const;
		RotatedBoxes operator[](torch::ArrayRef<torch::indexing::TensorIndex> item) const;

		/**
			Args:
				box_size (height, width): Size of the reference box covering
					[0, width] x [0, height]
				boundary_threshold (int): Boxes that extend beyond the reference box
					boundary by more than boundary_threshold are considered "outside".

			For RRPN, it might not be necessary to call this function since it's common
			for rotated box to extend to outside of the image boundaries
			(the clip function only clips the near-horizontal boxes)

			Returns:
				a binary vector, indicating whether each box is inside the reference box.
		*/
		virtual torch::Tensor inside_box(BoxSizeType box_size, int boundary_threshold = 0) const override;

		// The box centers in a Nx2 array of (x, y).
		virtual torch::Tensor get_centers() const override;

		/**
			Scale the rotated box with horizontal and vertical scaling factors
			Note: when scale_factor_x != scale_factor_y,
			the rotated box does not preserve the rectangular shape when the angle
			is not a multiple of 90 degrees under resize transformation.
			Instead, the shape is a parallelogram (that has skew)
			Here we make an approximation by fitting a rotated rectangle to the parallelogram.
		*/
		virtual void scale(float scale_x, float scale_y) override;
	};
}
