#pragma once

#include <Detectron2/Structures/Box2BoxTransform.h>
#include <Detectron2/Structures/Instances.h>
#include <Detectron2/Structures/ShapeSpec.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/roi_heads/fast_rcnn.py

	/**
		Call `fast_rcnn_inference_single_image` for all images.

		Args:
			boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
				boxes for each image. Element i has shape (Ri, K * 4) if doing
				class-specific regression, or (Ri, 4) if doing class-agnostic
				regression, where Ri is the number of predicted objects for image i.
				This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
			scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
				Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
				for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
			image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
			score_thresh (float): Only return detections with a confidence score exceeding this
				threshold.
			nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
			topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
				all detections.

		Returns:
			instances: (list[Instances]): A list of N instances, one for each image in the batch,
				that stores the topk most confidence detections.
			kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
				the corresponding boxes/scores index in [0, Ri) from the input, for image i.
	*/
	std::tuple<InstancesList, TensorVec> fast_rcnn_inference(const TensorVec &boxes, const TensorVec &scores,
		const std::vector<ImageSize> &image_shapes, float score_thresh, float nms_thresh, int topk_per_image);

	/**
		Single-image inference. Return bounding-box detection results by thresholding
		on scores and applying non-maximum suppression (NMS).

		Args:
			Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
			per image.

		Returns:
			Same as `fast_rcnn_inference`, but for only one image.
	*/
	std::tuple<InstancesPtr, torch::Tensor> fast_rcnn_inference_single_image(
		const torch::Tensor &boxes, const torch::Tensor &scores, const ImageSize &image_shape,
		float score_thresh, float nms_thresh, int topk_per_image);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// FastRCNNOutputLayers: Two linear layers for predicting Fast R-CNN outputs:
	//   (1) proposal-to-detection box regression deltas
	//   (2) classification scores
	class FastRCNNOutputLayersImpl : public torch::nn::Module {
	public:
		/**
			Args:
				input_shape (ShapeSpec): shape of the input feature to this module
				box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
				num_classes (int): number of foreground classes
				cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
				smooth_l1_beta (float): transition point from L1 to L2 loss.
				test_score_thresh (float): threshold to filter predictions results.
				test_nms_thresh (float): NMS threshold for prediction results.
				test_topk_per_image (int): number of top predictions to produce per image.
		*/
		FastRCNNOutputLayersImpl(CfgNode &cfg, const ShapeSpec &input_shape,
			const std::shared_ptr<Box2BoxTransform> &box2box_transform = nullptr);

		void initialize(const ModelImporter &importer, const std::string &prefix);

		// Tensor: Nx(K + 1) scores for each box
		// Tensor: Nx4 or Nx(Kx4) bounding box regression deltas.
		TensorVec forward(torch::Tensor x);

		// TODO: move the implementation to this class.
		/**
			Args:
				predictions: return values of :meth:`forward()`.
				proposals (list[Instances]): proposals that match the features
					that were used to compute predictions.
		*/
		TensorMap losses(const TensorVec &predictions, const InstancesList &proposals);

		/**
			Returns:
				list[Instances]: same as `fast_rcnn_inference`.
				list[Tensor]: same as `fast_rcnn_inference`.
		*/
		std::tuple<InstancesList, TensorVec> inference(const TensorVec &predictions, const InstancesList &proposals);

		/**
			Returns:
				list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
					class-specific box head. Element i of the list has shape (Ri, B), where Ri is
					the number of predicted objects for image i and B is the box dimension (4 or 5)
		*/
		TensorVec predict_boxes_for_gt_classes(const TensorVec &predictions, const InstancesList &proposals);

		/**
			Returns:
				list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
					for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
					the number of predicted objects for image i and B is the box dimension (4 or 5)
		*/
		TensorVec predict_boxes(const TensorVec &predictions, const InstancesList &proposals);

		/**
			Returns:
				list[Tensor]: A list of Tensors of predicted class probabilities for each image.
					Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
					for image i.
		*/
		TensorVec predict_probs(const TensorVec &predictions, const InstancesList &proposals);

	protected:
		std::shared_ptr<Box2BoxTransform> m_box2box_transform;
		torch::nn::Linear m_cls_score{ nullptr };
		torch::nn::Linear m_bbox_pred{ nullptr };
		float m_smooth_l1_beta;

	public:
		float m_test_score_thresh;
		float m_test_nms_thresh;
		int m_test_topk_per_image;
	};
	TORCH_MODULE(FastRCNNOutputLayers);
}