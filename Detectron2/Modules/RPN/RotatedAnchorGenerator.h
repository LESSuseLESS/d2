#pragma once

#include "AnchorGenerator.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/anchor_generator.py

	// RotatedAnchorGenerator: Computes rotated anchors used by Rotated RPN (RRPN),
	//   described in https://arxiv.org/abs/1703.01086 "Arbitrary-Oriented Scene Text Detection via Rotation Proposals"
	class RotatedAnchorGeneratorImpl : public AnchorGeneratorImpl {
	public:
		// angles: list of angles (in degrees CCW) to use for anchors. Same "broadcast" rule for `sizes` applies.
		RotatedAnchorGeneratorImpl(CfgNode &cfg, const ShapeSpec::Vec &input_shapes);
		RotatedAnchorGeneratorImpl(const std::vector<int> &strides, const std::vector<std::vector<float>> &sizes,
			const std::vector<std::vector<float>> &aspect_ratios, const std::vector<std::vector<float>> &angles,
			float offset = 0.5);

		virtual std::vector<int> num_anchors() const override;
		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;
		virtual BoxesList forward(const TensorVec &features) override;

	private:
		std::vector<std::vector<float>> m_sizes;
		std::vector<std::vector<float>> m_aspect_ratios;
		std::vector<std::vector<float>> m_angles;
		std::vector<int> m_strides;
		int m_num_features;
		float m_offset;

		// num_features of tensors of shape (len(sizes) * len(aspect_ratios) * len(angles), 5)
		//   storing anchor boxes in(x_ctr, y_ctr, w, h, angle) format.
		TensorVec m_cell_anchors;
	};
}