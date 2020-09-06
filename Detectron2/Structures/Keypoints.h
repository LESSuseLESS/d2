#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from structures/keypoints.py
	
	/**
		Stores keypoint annotation data. GT Instances have a `gt_keypoints` property
		containing the x,y location and visibility flag of each keypoint. This tensor has shape
		(N, K, 3) where N is the number of instances and K is the number of keypoints per instance.

		The visibility flag follows the COCO format and must be one of three integers:
		* v=0: not labeled (in which case x=y=0)
		* v=1: labeled but not visible
		* v=2: labeled and visible
	*/
	class Keypoints {
	public:
		// A wrapper around : func:`torch.nn.functional.interpolate` to support zero - size tensor.
		static torch::Tensor interpolate(const torch::Tensor &input,
			const torch::nn::functional::InterpolateFuncOptions &options);

		// TODO make this nicer, this is a direct translation from C2 (but removing the inner loop)
		/**
			Encode keypoint locations into a target heatmap for use in SoftmaxWithLoss across space.

			Maps keypoints from the half-open interval [x1, x2) on continuous image coordinates to the
			closed interval [0, heatmap_size - 1] on discrete image coordinates. We use the
			continuous-discrete conversion from Heckbert 1990 ("What is the coordinate of a pixel?"):
			d = floor(c) and c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.

			Arguments:
				keypoints: tensor of keypoint locations in of shape (N, K, 3).
				rois: Nx4 tensor of rois in xyxy format
				heatmap_size: integer side length of square heatmap.

			Returns:
				heatmaps: A tensor of shape (N, K) containing an integer spatial label
					in the range [0, heatmap_size**2 - 1] for each keypoint in the input.
				valid: A tensor of shape (N, K) containing whether each keypoint is in
					the roi or not.
		*/
		static std::tuple<torch::Tensor, torch::Tensor>
			_keypoints_to_heatmap(torch::Tensor keypoints, torch::Tensor rois, int heatmap_size);

		/**
			Extract predicted keypoint locations from heatmaps.

			Args:
				maps (Tensor): (#ROIs, #keypoints, POOL_H, POOL_W). The predicted heatmap of logits for
					each ROI and each keypoint.
				rois (Tensor): (#ROIs, 4). The box of each ROI.

			Returns:
				Tensor of shape (#ROIs, #keypoints, 4) with the last dimension corresponding to
				(x, y, logit, score) for each keypoint.

			When converting discrete pixel indices in an NxN image to a continuous keypoint coordinate,
			we maintain consistency with :meth:`Keypoints.to_heatmap` by using the conversion from
			Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.
		*/
		static torch::Tensor heatmaps_to_keypoints(torch::Tensor maps, torch::Tensor rois);

	public:
		/**
			keypoints: A Tensor, numpy array, or list of the x, y, and visibility of each keypoint.
				The shape should be (N, K, 3) where N is the number of
				instances, and K is the number of keypoints per instance.
		*/
		Keypoints(const torch::Tensor &keypoints);

		int len() const { return m_tensor.size(0); }
		torch::Device device() const { return m_tensor.device(); }
		std::string toString() const;

		torch::Tensor tensor() const { return m_tensor; }

		/**
			Arguments:
				boxes: Nx4 tensor, the boxes to draw the keypoints to

			Returns:
				heatmaps:
					A tensor of shape (N, K) containing an integer spatial label
					in the range [0, heatmap_size**2 - 1] for each keypoint in the input.
				valid:
					A tensor of shape (N, K) containing whether each keypoint is in the roi or not.
		*/
		std::tuple<torch::Tensor, torch::Tensor> to_heatmap(torch::Tensor boxes, int heatmap_size);

		/**
			Create a new `Keypoints` by indexing on this `Keypoints`.

			The following usage are allowed:

			1. `new_kpts = kpts[3]`: return a `Keypoints` which contains only one instance.
			2. `new_kpts = kpts[2:10]`: return a slice of key points.
			3. `new_kpts = kpts[vector]`, where vector is a torch.ByteTensor
			   with `length = len(kpts)`. Nonzero elements in the vector will be selected.

			Note that the returned Keypoints might share storage with this Keypoints,
			subject to Pytorch's indexing semantics.
		*/
		Keypoints operator[](int64_t item) const;
		Keypoints operator[](torch::ArrayRef<torch::indexing::TensorIndex> item) const;

	private:
		torch::Tensor m_tensor;
	};
}