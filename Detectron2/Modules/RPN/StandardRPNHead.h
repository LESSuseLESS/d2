#pragma once

#include <Detectron2/Modules/Conv/ConvBn2d.h>
#include "AnchorGenerator.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/proposal_generator/rpn.py

	// RPN heads, which take feature maps and perform objectness classification and bounding box regression for anchors
	//
	// Standard RPN classification and regression heads described in :paper:`Faster R-CNN`. Uses a 3x3 conv to produce
	// a shared hidden state from which one 1x1 conv predicts objectness logits for each anchor and a second 1x1 conv
	// predicts bounding - box deltas specifying how to deform each anchor into an object proposal.
	struct StandardRPNHeadImpl : public torch::nn::Module {
	public:
		// in_channels: number of input feature channels. When using multiple input features, they must have the
		//   same number of channels.
		// num_anchors: number of anchors to predict for *each spatial position* on the feature map. The total number
		//   of anchors for each feature map will be `num_anchors * H * W`.
		// box_dim: dimension of a box, which is also the number of box regression predictions to make for each anchor.
		//   An axis aligned box has box_dim=4, while a rotated box has box_dim=5.
		StandardRPNHeadImpl(CfgNode &cfg, const ShapeSpec::Vec &input_shapes);
		void initialize(const ModelImporter &importer, const std::string &prefix);

		// features: list of feature maps
		// Returns:
		//   list[Tensor]: A list of L elements.
		//       Element i is a tensor of shape (N, A, Hi, Wi) representing
		//       the predicted objectness logits for all anchors. A is the number of cell anchors.
		//   list[Tensor]: A list of L elements. Element i is a tensor of shape
		//       (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
		//       to proposals.
		std::vector<TensorVec> forward(const TensorVec &features);

	private:
		ConvBn2d m_conv{ nullptr };					// 3x3 conv for the hidden representation
		ConvBn2d m_objectness_logits{ nullptr };	// 1x1 conv for predicting objectness logits
		ConvBn2d m_anchor_deltas{ nullptr };		// 1x1 conv for predicting box2box transform deltas
	};
	TORCH_MODULE(StandardRPNHead);
	using RPNHead = StandardRPNHead;

	// Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
	RPNHead build_rpn_head(CfgNode &cfg, const ShapeSpec::Vec &input_shapes);
}