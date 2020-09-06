#include "Base.h"
#include "MaskRCNNConvUpsampleHead.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MaskRCNNConvUpsampleHeadImpl::MaskRCNNConvUpsampleHeadImpl(CfgNode &cfg, const ShapeSpec &input_shape) :
	BaseMaskRCNNHeadImpl(cfg),
	m_num_classes(cfg["MODEL.ROI_HEADS.NUM_CLASSES"].as<int>())
{
	vector<int64_t> conv_dims;
	{
		auto conv_dim = cfg["MODEL.ROI_MASK_HEAD.CONV_DIM"].as<int>();
		auto num_conv = cfg["MODEL.ROI_MASK_HEAD.NUM_CONV"].as<int>();
		conv_dims = std::vector<int64_t>(num_conv + 1, conv_dim);	// +1 for ConvTranspose
	}
	auto conv_norm = BatchNorm::GetType(cfg["MODEL.ROI_MASK_HEAD.NORM"].as<string>());
	bool cls_agnostic_mask = cfg["MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK"].as<bool>();
	if (cls_agnostic_mask) {
		m_num_classes = 1;
	}

	assert(conv_dims.size() >= 1); // conv_dims have to be non-empty!
	int cur_channels = input_shape.channels;
	for (int k = 0; k < conv_dims.size() - 1; k++) {
		auto conv_dim = conv_dims[k];
		auto conv = ConvBn2d(nn::Conv2dOptions(cur_channels, conv_dim, 3).stride(1).padding(1), conv_norm, true);
		register_module(FormatString("mask_fcn%d", k + 1), conv);
		m_conv_norm_relus.push_back(conv);
		cur_channels = conv_dim;
	}
	
	auto last_conv_dim = conv_dims[conv_dims.size() - 1];
	m_deconv = nn::ConvTranspose2d(nn::ConvTranspose2dOptions(cur_channels, last_conv_dim, 2).stride(2).padding(0));
	register_module("deconv", m_deconv);
	cur_channels = last_conv_dim;

	m_predictor = ConvBn2d(nn::Conv2dOptions(cur_channels, m_num_classes, 1).stride(1).padding(0));
	register_module("predictor", m_predictor);
}

void MaskRCNNConvUpsampleHeadImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	for (int i = 0; i < m_conv_norm_relus.size(); i++) {
		m_conv_norm_relus[i]->initialize(importer, prefix + FormatString(".mask_fcn%d", i + 1),
			ModelImporter::kCaffe2MSRAFill);
	}
	importer.Import(prefix + ".deconv", m_deconv, ModelImporter::kCaffe2MSRAFill);

	// use normal distribution initialization for mask prediction layer
	m_predictor->initialize(importer, prefix + ".predictor", ModelImporter::kNormalFill3);
}

torch::Tensor MaskRCNNConvUpsampleHeadImpl::layers(torch::Tensor x) {
	for (auto &layer : m_conv_norm_relus) {
		x = layer(x);
	}
	x = relu(m_deconv(x));
	return m_predictor(x);
}

