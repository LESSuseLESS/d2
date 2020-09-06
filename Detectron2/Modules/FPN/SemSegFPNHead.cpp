#include "Base.h"
#include "SemSegFPNHead.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

SemSegFPNHeadImpl::SemSegFPNHeadImpl(CfgNode &cfg, const ShapeSpec::Map &input_shapes) :
	m_in_features(cfg["MODEL.SEM_SEG_HEAD.IN_FEATURES"].as<vector<string>>()),
	m_ignore_value(cfg["MODEL.SEM_SEG_HEAD.IGNORE_VALUE"].as<int>()),
	m_common_stride(cfg["MODEL.SEM_SEG_HEAD.COMMON_STRIDE"].as<int>()),
	m_loss_weight(cfg["MODEL.SEM_SEG_HEAD.LOSS_WEIGHT"].as<float>())
{
	// Number of classes in the semantic segmentation head
	auto num_classes = cfg["MODEL.SEM_SEG_HEAD.NUM_CLASSES"].as<int>();
	// Number of channels in the 3x3 convs inside semantic-FPN heads.
	auto conv_dims = cfg["MODEL.SEM_SEG_HEAD.CONVS_DIM"].as<int>();
	// Normalization method for the convolution layers. Options: "" (no norm), "GN".
	auto norm = BatchNorm::GetType(cfg["MODEL.SEM_SEG_HEAD.NORM"].as<string>());

	auto upsample_options = nn::UpsampleOptions()
		.scale_factor(vector<double>{ 2, 2 })
		.mode(torch::kBilinear)
		.align_corners(false);

	m_interpolate_options
		.scale_factor(vector<double>{ (double)m_common_stride, (double)m_common_stride })
		.mode(torch::kBilinear)
		.align_corners(false);

	bool bias = (norm == BatchNorm::kNone);
	m_scale_heads.reserve(m_in_features.size());
	for (const auto &in_feature : m_in_features) {
		auto iter = input_shapes.find(in_feature);
		assert(iter != input_shapes.end());
		auto &shape = iter->second;

		nn::Sequential head_ops;
		auto head_length = max(1, IntLog2(shape.stride) - IntLog2(m_common_stride));
		for (int k = 0; k < head_length; k++) {
			int in_channels = (k == 0 ? shape.channels : conv_dims);
			auto options = nn::Conv2dOptions(in_channels, conv_dims, 3).stride(1).padding(1).bias(bias);
			ConvBn2d conv(options, norm, true);
			head_ops->push_back(conv);
			if (shape.stride != m_common_stride) {
				head_ops->push_back(nn::Upsample(upsample_options));
			}
		}
		register_module(in_feature, head_ops);
		m_scale_heads.push_back(head_ops);
	}

	m_predictor = ConvBn2d(nn::Conv2dOptions(conv_dims, num_classes, 1).stride(1).padding(0));
	register_module("predictor", m_predictor);
}

void SemSegFPNHeadImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	for (int i = 0; i < m_scale_heads.size(); i++) {
		auto seq = m_scale_heads[i];
		for (int k = 0; k < seq->size(); k++) {
			auto m = dynamic_pointer_cast<ConvBn2dImpl>(seq[k]);
			if (m) {
				auto name = prefix + "." + m_in_features[i] + FormatString(".%d", k);
				m->initialize(importer, name, ModelImporter::kCaffe2MSRAFill);
			}
		}
	}
	m_predictor->initialize(importer, prefix + ".predictor", ModelImporter::kCaffe2MSRAFill);
}

std::tuple<torch::Tensor, TensorMap> SemSegFPNHeadImpl::forward(const TensorMap &features, const Tensor &targets) {
	auto x = layers(features);
	if (is_training()) {
		return { Tensor(), losses(x, targets) };
	}
	x = nn::functional::interpolate(x, m_interpolate_options);
	return { x, {} };
}

torch::Tensor SemSegFPNHeadImpl::layers(const TensorMap &features) {
	Tensor x;
	for (int i = 0; i < m_in_features.size(); i++) {
		auto &f = m_in_features[i];
		auto iter = features.find(f);
		assert(iter != features.end());

		if (i == 0) {
			x = m_scale_heads[i]->forward(iter->second);
		}
		else {
			x = x + m_scale_heads[i]->forward(iter->second);
		}
	}
	x = m_predictor(x);
	return x;
}

TensorMap SemSegFPNHeadImpl::losses(torch::Tensor predictions, torch::Tensor targets) {
	predictions = nn::functional::interpolate(predictions, m_interpolate_options);
	auto loss = nn::functional::cross_entropy(predictions, targets,
		nn::functional::CrossEntropyFuncOptions()
		.reduction(torch::kMean).ignore_index(m_ignore_value));

	return { { "loss_sem_seg", loss * m_loss_weight } };
}
