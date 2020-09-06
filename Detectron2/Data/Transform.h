#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from fvcore/transforms/transform.py
	
    /**
		Base class for implementations of **deterministic** transformations for
		image and other data structures. "Deterministic" requires that the output
		of all methods of this class are deterministic w.r.t their input arguments.
		Note that this is different from (random) data augmentations. To perform
		data augmentations in training, there should be a higher-level policy that
		generates these transform ops.

		Each transform op may handle several data types, e.g.: image, coordinates,
		segmentation, bounding boxes, with its ``apply_*`` methods. Some of
		them have a default implementation, but can be overwritten if the default
		isn't appropriate. See documentation of each pre-defined ``apply_*`` methods
		for details. Note that The implementation of these method may choose to
		modify its input data in-place for efficient transformation.

		The class can be extended to support arbitrary new data types with its
		:meth:`register_type` method.
	*/
	class Transform : public std::enable_shared_from_this<Transform> {
	public:
		enum Interp {
			kNone		= cv::InterpolationFlags::INTER_NEAREST,
			kNEAREST	= cv::InterpolationFlags::INTER_NEAREST,
			kBILINEAR	= cv::InterpolationFlags::INTER_LINEAR,
			kBICUBIC	= cv::InterpolationFlags::INTER_CUBIC
		};

	public:
		virtual ~Transform() {}

		/**
			Set attributes from the input list of parameters.

			Args:
				params (list): list of parameters.
		*/
		void _set_attributes(const std::unordered_map<std::string, YAML::Node> &params);

		/**
			Apply the transform on an image.

			Args:
				img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
					of type uint8 in range [0, 255], or floating point in range
					[0, 1] or [0, 255].
			Returns:
				ndarray: image after apply the transformation.
		*/
		virtual torch::Tensor apply_image(torch::Tensor img, Interp interp = kNone) = 0;

		/**
			Apply the transform on coordinates.

			Args:
				coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

			Returns:
				ndarray: coordinates after apply the transformation.

			Note:
				The coordinates are not pixel indices. Coordinates inside an image of
				shape (H, W) are in range [0, W] or [0, H].
				This function should correctly transform coordinates outside the image as well.
		*/
		virtual torch::Tensor apply_coords(torch::Tensor coords) = 0;

		/**
			Apply the transform on a full-image segmentation.
			By default will just perform "apply_image".

			Args:
				segmentation (ndarray): of shape HxW. The array should have integer
				or bool dtype.

			Returns:
				ndarray: segmentation after apply the transformation.
		*/
		virtual torch::Tensor apply_segmentation(torch::Tensor segmentation) {
			return apply_image(segmentation);
		}

		/**
			Apply the transform on an axis-aligned box. By default will transform
			the corner points and use their minimum/maximum to create a new
			axis-aligned box. Note that this default may change the size of your
			box, e.g. after rotations.

			Args:
				box (ndarray): Nx4 floating point array of XYXY format in absolute
					coordinates.
			Returns:
				ndarray: box after apply the transformation.

			Note:
				The coordinates are not pixel indices. Coordinates inside an image of
				shape (H, W) are in range [0, W] or [0, H].

				This function does not clip boxes to force them inside the image.
				It is up to the application that uses the boxes to decide.
		*/
		virtual torch::Tensor apply_box(torch::Tensor box);

		/**
			Apply the transform on a list of polygons, each represented by a Nx2
			array. By default will just transform all the points.

			Args:
				polygon (list[ndarray]): each is a Nx2 floating point array of
					(x, y) format in absolute coordinates.
			Returns:
				list[ndarray]: polygon after apply the transformation.

			Note:
				The coordinates are not pixel indices. Coordinates on an image of
				shape (H, W) are in range [0, W] or [0, H].
		*/
		virtual TensorVec apply_polygons(const TensorVec &polygons);

		/**
			Register the given function as a handler that this transform will use
			for a specific data type.

			Args:
				data_type (str): the name of the data type (e.g., box)
				func (callable): takes a transform and a data, returns the
					transformed data.

			Examples:

			.. code-block:: python

				# call it directly
				def func(flip_transform, voxel_data):
					return transformed_voxel_data
				HFlipTransform.register_type("voxel", func)

				# or, use it as a decorator
				@HFlipTransform.register_type("voxel")
				def func(flip_transform, voxel_data):
					return transformed_voxel_data

				# ...
				transform = HFlipTransform(...)
				transform.apply_voxel(voxel_data)  # func will be called
		*/
		//static void register_type(const std::string &data_type /*, func: Optional[Callable] = None */);

		/**
			Create a transform that inverts the geometric changes (i.e. change of
			coordinates) of this transform.

			Note that the inverse is meant for geometric changes only.
			The inverse of photometric transforms that do not change coordinates
			is defined to be a no-op, even if they may be invertible.

			Returns:
				Transform:
		*/
		virtual std::shared_ptr<Transform> inverse();

		/**
			Produce something like:
			"MyTransform(field1={self.field1}, field2={self.field2})"
		*/
		std::string repr() const;

	private:
		std::unordered_map<std::string, YAML::Node> m_attrs;
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// A transform that does nothing.
	class NoOpTransform : public Transform {
	public:
		virtual torch::Tensor apply_image(torch::Tensor img, Interp interp = kNone) override {
			return img;
		}
		virtual torch::Tensor apply_coords(torch::Tensor coords) override {
			return coords;
		}
		virtual std::shared_ptr<Transform> inverse() override {
			return shared_from_this();
		}
	};
}
