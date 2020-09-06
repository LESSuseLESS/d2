#include "Base.h"
#include "KRCNNConvDeconvUpsampleHead.h"

#include <Detectron2/Structures/Keypoints.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

KRCNNConvDeconvUpsampleHeadImpl::KRCNNConvDeconvUpsampleHeadImpl(CfgNode &cfg, const ShapeSpec &input_shape) :
	BaseKeypointRCNNHeadImpl(cfg)
{
	auto conv_dims = cfg["MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS"].as<vector<int64_t>>();

	// default up_scale to 2 (this can be made an option)
	int up_scale = 2;
	int in_channels = input_shape.channels;

	for (int idx = 0; idx < conv_dims.size(); idx++) {
		auto &layer_channels = conv_dims[idx];
		auto module = ConvBn2d(nn::Conv2dOptions(in_channels, layer_channels, 3).stride(1).padding(1));
		register_module(FormatString("conv_fcn%d", idx + 1), module);
		m_blocks.push_back(module);
		in_channels = layer_channels;
	}

	int deconv_kernel = 4;
	m_score_lowres = nn::ConvTranspose2d(nn::ConvTranspose2dOptions(in_channels, m_num_keypoints, deconv_kernel)
		.stride(2).padding(deconv_kernel / 2 - 1));
	register_module("score_lowres", m_score_lowres);
	m_up_scale = up_scale;
}

void KRCNNConvDeconvUpsampleHeadImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	for (int i = 0; i < m_blocks.size(); i++) {
		m_blocks[i]->initialize(importer, prefix + FormatString(".conv_fcn%d", i + 1), ModelImporter::kCaffe2MSRAFill);
	}
	importer.Import(prefix + ".score_lowres", m_score_lowres, ModelImporter::kCaffe2MSRAFill);
}

torch::Tensor KRCNNConvDeconvUpsampleHeadImpl::layers(torch::Tensor x) {
	for (auto &layer : m_blocks) {
		x = relu(layer(x));
	}
	x = m_score_lowres(x);
	auto options = nn::functional::InterpolateFuncOptions()
		.scale_factor(vector<double>{ (double)m_up_scale, (double)m_up_scale })
		.mode(torch::kBilinear).align_corners(false);
	x = Keypoints::interpolate(x, options);
	return x;
}