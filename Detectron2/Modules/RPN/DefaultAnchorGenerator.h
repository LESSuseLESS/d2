#pragma once

#include "AnchorGenerator.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/anchor_generator.py

	// DefaultAnchorGenerator: Computes anchors in the standard ways described in https://arxiv.org/abs/1506.01497
	class DefaultAnchorGeneratorImpl : public AnchorGeneratorImpl {
	public:
		/**
			sizes: list of anchor sizes (i.e. sqrt of anchor area) to use for the i-th feature map. Anchor sizes are
				given in absolute lengths in units of the input image; they do not dynamically scale if input image
				size changes.
			aspect_ratios: list of aspect ratios (i.e. height / width) to use for anchors. Same "broadcast" rule for
				`sizes` applies.
			strides: stride of each input feature.
			offset: Relative offset between the center of the first anchor and the top-left corner of the image. Value
				has to be in [0, 1). Recommend to use 0.5, which means half stride.
		*/
		DefaultAnchorGeneratorImpl(CfgNode &cfg, const ShapeSpec::Vec &input_shapes);
		DefaultAnchorGeneratorImpl(const std::vector<int> &strides, const std::vector<std::vector<float>> &sizes,
			const std::vector<std::vector<float>> &aspect_ratios, float offset = 0.5);

		virtual std::vector<int> num_anchors() const override;
		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;
		virtual BoxesList forward(const TensorVec &features) override;

	private:
		std::vector<std::vector<float>> m_sizes;
		std::vector<std::vector<float>> m_aspect_ratios;
		std::vector<int> m_strides;
		int m_num_features;
		float m_offset;

		// num_features of tensors of shape(len(sizes) * len(aspect_ratios), 4) storing anchor boxes in XYXY format.
		TensorVec m_cell_anchors;
	};
}