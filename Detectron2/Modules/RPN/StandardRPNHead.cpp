#include "Base.h"
#include "StandardRPNHead.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RPNHead Detectron2::build_rpn_head(CfgNode &cfg, const ShapeSpec::Vec &input_shapes) {
	auto name = cfg["MODEL.RPN.HEAD_NAME"].as<string>();
	if (name == "StandardRPNHead") {
		return make_shared<StandardRPNHeadImpl>(cfg, input_shapes);
	}
	assert(false);
	return nullptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

StandardRPNHeadImpl::StandardRPNHeadImpl(CfgNode &cfg, const ShapeSpec::Vec &input_shapes) {
	auto anchor_generator = build_anchor_generator(cfg, input_shapes);
	int box_dim = anchor_generator->box_dim();
	auto anchors = anchor_generator->num_anchors();
	int num_anchors = anchors[0];
	for (int i = 1; i < anchors.size(); i++) {
		assert(anchors[i] == num_anchors);
	}

	auto in_channels = ShapeSpec::channels_single(input_shapes);
	m_conv = ConvBn2d(nn::Conv2dOptions(in_channels, in_channels, 3).padding(1));
	register_module("conv", m_conv);
	m_objectness_logits = ConvBn2d(nn::Conv2dOptions(in_channels, num_anchors, 1));
	register_module("objectness_logits", m_objectness_logits);
	m_anchor_deltas = ConvBn2d(nn::Conv2dOptions(in_channels, num_anchors * box_dim, 1));
	register_module("anchor_deltas", m_anchor_deltas);
}

void StandardRPNHeadImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	m_conv->initialize(importer, prefix + ".conv", ModelImporter::kNormalFill2);
	m_objectness_logits->initialize(importer, prefix + ".objectness_logits", ModelImporter::kNormalFill2);
	m_anchor_deltas->initialize(importer, prefix + ".anchor_deltas", ModelImporter::kNormalFill2);
}

vector<TensorVec> StandardRPNHeadImpl::forward(const TensorVec &features) {
	TensorVec pred_objectness_logits;
	TensorVec pred_anchor_deltas;
	for (auto x : features) {
		x = relu(m_conv(x));
		pred_objectness_logits.push_back(m_objectness_logits(x));
		pred_anchor_deltas.push_back(m_anchor_deltas(x));
	}
	return { pred_objectness_logits, pred_anchor_deltas };
}
