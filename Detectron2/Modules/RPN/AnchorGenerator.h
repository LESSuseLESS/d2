#pragma once

#include <Detectron2/Structures/ShapeSpec.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/anchor_generator.py

	// AnchorGenerator: Creates object detection anchors for feature maps.
	class AnchorGeneratorImpl : public torch::nn::Module {
	public:
		static TensorVec _create_grid_offsets(const std::pair<int, int> &size, int stride, float offset,
			torch::Device device);

		// If one size (or aspect ratio) is specified and there are multiple feature maps, we "broadcast" anchors
		// of that single size (or aspect ratio) over all feature maps.
		// If params is list[float], or list[list[float]] with len(params) == 1, repeat it num_features time.
		// Returns: list[list[float]]: param for each feature
		static std::vector<std::vector<float>> _broadcast_params(const std::vector<std::vector<float>> &params,
			int num_features);
		static std::vector<std::vector<float>> _broadcast_params(const std::vector<float> &params,
			int num_features);

	public:
		AnchorGeneratorImpl(int box_dim) : m_box_dim(box_dim) {}
		virtual ~AnchorGeneratorImpl() {}

		int box_dim() const { return m_box_dim; }

		// Returns: Each int is the number of anchors at every pixel location, on that feature map. For example,
		//   if at every pixel we use anchors of 3 aspect ratios and 5 sizes, the number of anchors is 15.
		// In standard RPN models, `num_anchors` on every feature map is the same.
		virtual std::vector<int> num_anchors() const = 0;

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) = 0;

		// features: list of backbone feature maps on which to generate anchors.
		// Returns: a list of Boxes containing all the anchors for each feature map (i.e.the cell anchors repeated
		//   over all locations in the feature map). The number of anchors of each feature map is
		//   Hi x Wi x num_cell_anchors, where Hi, Wi are resolution of the feature map divided by anchor stride.
		virtual BoxesList forward(const TensorVec &features) = 0;

	protected:
		int m_box_dim; // the dimension of each anchor box.

		// this was done by BufferList in original coding:
		void register_cell_anchors(const TensorVec &cell_anchors);
	};
	TORCH_MODULE(AnchorGenerator);

	// Built an anchor generator from `cfg.MODEL.ANCHOR_GENERATOR.NAME`.
	AnchorGenerator build_anchor_generator(CfgNode &cfg, const ShapeSpec::Vec &input_shapes);
}