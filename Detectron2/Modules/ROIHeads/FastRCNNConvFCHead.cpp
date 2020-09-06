#include "Base.h"
#include "FastRCNNConvFCHead.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BoxHead Detectron2::build_box_head(CfgNode &cfg, const ShapeSpec &input_shape) {
	return shared_ptr<FastRCNNConvFCHeadImpl>(new FastRCNNConvFCHeadImpl(cfg, input_shape));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FastRCNNConvFCHeadImpl::FastRCNNConvFCHeadImpl(CfgNode &cfg, const ShapeSpec &input_shape) {
	// Channel dimension for Conv layers in the RoI box head
	auto conv_dim = cfg["MODEL.ROI_BOX_HEAD.CONV_DIM"].as<int>();
	auto num_conv = cfg["MODEL.ROI_BOX_HEAD.NUM_CONV"].as<int>();

	// Hidden layer dimension for FC layers in the RoI box head
	auto fc_dim = cfg["MODEL.ROI_BOX_HEAD.FC_DIM"].as<int>();
	auto num_fc = cfg["MODEL.ROI_BOX_HEAD.NUM_FC"].as<int>();

	// Normalization method for the convolution layers. Options: "" (no norm), "GN", "SyncBN".
	auto conv_norm = BatchNorm::GetType(cfg["MODEL.ROI_BOX_HEAD.NORM"].as<string>());

	vector<int> conv_dims(num_conv, conv_dim);	// the output dimensions of the conv layers
	vector<int> fc_dims(num_fc, fc_dim);		// the output dimensions of the fc layers
	assert(num_conv + num_fc > 0);

	m_output_size = input_shape;
	bool bias = (conv_norm == BatchNorm::kNone);
	for (int k = 0; k < conv_dims.size(); k++) {
		auto conv_dim = conv_dims[k];
		ConvBn2d conv(nn::Conv2dOptions(m_output_size.channels, conv_dim, 3).padding(1).bias(bias),
			conv_norm, true);
		register_module(FormatString("conv%d", k + 1), conv);
		m_conv_norm_relus.push_back(conv);
		m_output_size.channels = conv_dim;
	}

	for (int k = 0; k < fc_dims.size(); k++) {
		auto fc_dim = fc_dims[k];
		auto fc = torch::nn::Linear(m_output_size.prod(), fc_dim);
		register_module(FormatString("fc%d", k + 1), fc);
		m_fcs.push_back(fc);
		m_output_size.channels = fc_dim;
		m_output_size.height = m_output_size.width = 1;
	}
}

void FastRCNNConvFCHeadImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	for (int i = 0; i < m_conv_norm_relus.size(); i++) {
		m_conv_norm_relus[i]->initialize(importer, prefix + FormatString(".conv%d", i + 1),
			ModelImporter::kCaffe2MSRAFill);
	}
	for (int i = 0; i < m_fcs.size(); i++) {
		importer.Import(prefix + FormatString(".fc%d", i + 1), m_fcs[i], ModelImporter::kCaffe2XavierFill);
	}
}

torch::Tensor FastRCNNConvFCHeadImpl::forward(torch::Tensor x) {
	for (auto conv : m_conv_norm_relus) {
		x = conv(x);
	}
	if (!m_fcs.empty()) {
		if (x.dim() > 2) {
			x = torch::flatten(x, 1);
		}
		for (auto fc : m_fcs) {
			x = fc(x);
			x = relu(x);
		}
	}
	return x;
}
