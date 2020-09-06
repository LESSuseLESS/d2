#include "Base.h"
#include "Box2BoxTransform.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const float Box2BoxTransform::_DEFAULT_SCALE_CLAMP = log(1000.0 / 16);

std::shared_ptr<Box2BoxTransform> Box2BoxTransform::Create(YAML::Node node, float scale_clamp) {
	auto w = node.as<vector<float>>();
	return Create(w, scale_clamp);
}

std::shared_ptr<Box2BoxTransform> Box2BoxTransform::Create(const vector<float> &w, float scale_clamp) {
	Weights weights{ w[0], w[1], w[2], w[3] };
	switch (w.size()) {
	case 4:
		return make_shared<Box2BoxTransform>(weights, scale_clamp);
	case 5:
		weights.da = w[4];
		return make_shared<Box2BoxTransformRotated>(weights, scale_clamp);
	}
	assert(false);
	return nullptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Box2BoxTransform::Box2BoxTransform(const Weights &weights, float scale_clamp) :
	m_weights(weights), m_scale_clamp(scale_clamp) {
}

torch::Tensor Box2BoxTransform::apply_deltas_broadcast(torch::Tensor deltas, torch::Tensor boxes) {
	assert(deltas.dim() == 2 && boxes.dim() == 2);
	auto N = boxes.size(0);
	auto B = boxes.size(1);
	assert(deltas.size(1) % B == 0); // Second dim of deltas should be a multiple of {B}. Got {deltas.shape}
	auto K = deltas.size(1) / B;
	auto ret = apply_deltas(
		deltas.view({ N * K, B }), boxes.unsqueeze(1).expand({ N, K, B }).reshape({ N * K, B })
	);
	return ret.view({ N, K * B });
}

torch::Tensor Box2BoxTransform::get_deltas(torch::Tensor src_boxes, torch::Tensor target_boxes) {
	auto src_widths = src_boxes.index({ Colon, 2 }) - src_boxes.index({ Colon, 0 });
	auto src_heights = src_boxes.index({ Colon, 3 }) - src_boxes.index({ Colon, 1 });
	auto src_ctr_x = src_boxes.index({ Colon, 0 }) + 0.5 * src_widths;
	auto src_ctr_y = src_boxes.index({ Colon, 1 }) + 0.5 * src_heights;

	auto target_widths = target_boxes.index({ Colon, 2 }) - target_boxes.index({ Colon, 0 });
	auto target_heights = target_boxes.index({ Colon, 3 }) - target_boxes.index({ Colon, 1 });
	auto target_ctr_x = target_boxes.index({ Colon, 0 }) + 0.5 * target_widths;
	auto target_ctr_y = target_boxes.index({ Colon, 1 }) + 0.5 * target_heights;

	auto dx = m_weights.dx * (target_ctr_x - src_ctr_x) / src_widths;
	auto dy = m_weights.dy * (target_ctr_y - src_ctr_y) / src_heights;
	auto dw = m_weights.dw * torch::log(target_widths / src_widths);
	auto dh = m_weights.dh * torch::log(target_heights / src_heights);

	auto deltas = torch::stack({ dx, dy, dw, dh }, 1);
	assert((src_widths > 0).all().item<bool>()); // Input boxes to Box2BoxTransform are not valid!
	return deltas;
}

torch::Tensor Box2BoxTransform::apply_deltas(torch::Tensor deltas, torch::Tensor boxes) {
	boxes = boxes.to(deltas.dtype());

	auto widths = boxes.index({ Colon, 2 }) - boxes.index({ Colon, 0 });
	auto heights = boxes.index({ Colon, 3 }) - boxes.index({ Colon, 1 });
	auto ctr_x = boxes.index({ Colon, 0 }) + 0.5 * widths;
	auto ctr_y = boxes.index({ Colon, 1 }) + 0.5 * heights;

	auto dx = deltas.index({ Colon, Slice(0, None, 4) }) / m_weights.dx;
	auto dy = deltas.index({ Colon, Slice(1, None, 4) }) / m_weights.dy;
	auto dw = deltas.index({ Colon, Slice(2, None, 4) }) / m_weights.dw;
	auto dh = deltas.index({ Colon, Slice(3, None, 4) }) / m_weights.dh;

	// Prevent sending too large values into torch.exp()
	dw = torch::clamp(dw, nullopt, m_scale_clamp);
	dh = torch::clamp(dh, nullopt, m_scale_clamp);

	vector<torch::indexing::TensorIndex> sliceAllNone{ Colon, None };
	auto pred_ctr_x = dx * widths.index(sliceAllNone) + ctr_x.index(sliceAllNone);
	auto pred_ctr_y = dy * heights.index(sliceAllNone) + ctr_y.index(sliceAllNone);
	auto pred_w = torch::exp(dw) * widths.index(sliceAllNone);
	auto pred_h = torch::exp(dh) * heights.index(sliceAllNone);

	auto pred_boxes = torch::zeros_like(deltas);
	pred_boxes.index_put_({ Colon, Slice(0, None, 4) }, pred_ctr_x - 0.5 * pred_w);  // x1
	pred_boxes.index_put_({ Colon, Slice(1, None, 4) }, pred_ctr_y - 0.5 * pred_h);  // y1
	pred_boxes.index_put_({ Colon, Slice(2, None, 4) }, pred_ctr_x + 0.5 * pred_w);  // x2
	pred_boxes.index_put_({ Colon, Slice(3, None, 4) }, pred_ctr_y + 0.5 * pred_h);  // y2
	return pred_boxes;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Box2BoxTransformRotated::Box2BoxTransformRotated(const Weights &weights, float scale_clamp) :
	Box2BoxTransform(weights, scale_clamp) {
}

torch::Tensor Box2BoxTransformRotated::get_deltas(torch::Tensor src_boxes, torch::Tensor target_boxes) {
	auto srcs = src_boxes.unbind(1);
	auto src_ctr_x = srcs[0];
	auto src_ctr_y = srcs[1];
	auto src_widths = srcs[2];
	auto src_heights = srcs[3];
	auto src_angles = srcs[4];

	auto targets = target_boxes.unbind(1);
	auto target_ctr_x = targets[0];
	auto target_ctr_y = targets[1];
	auto target_widths = targets[2];
	auto target_heights = targets[3];
	auto target_angles = targets[4];

	auto dx = m_weights.dx * (target_ctr_x - src_ctr_x) / src_widths;
	auto dy = m_weights.dy * (target_ctr_y - src_ctr_y) / src_heights;
	auto dw = m_weights.dw * torch::log(target_widths / src_widths);
	auto dh = m_weights.dh * torch::log(target_heights / src_heights);
	// Angles of deltas are in radians while angles of boxes are in degrees.
	// the conversion to radians serve as a way to normalize the values
	auto da = target_angles - src_angles;
	da = (da + 180.0) % 360.0 - 180.0;  // make it in [-180, 180)
	da *= m_weights.da * M_PI / 180.0;

	auto deltas = torch::stack({ dx, dy, dw, dh, da }, 1);
	assert((src_widths > 0).all().item<bool>()); // Input boxes to Box2BoxTransformRotated are not valid!
	return deltas;
}

torch::Tensor Box2BoxTransformRotated::apply_deltas(torch::Tensor deltas, torch::Tensor boxes) {
	assert(deltas.size(1) == 5 && boxes.size(1) == 5);

	boxes = boxes.to(deltas.dtype());

	auto ctr_x = boxes.index({ Colon, 0 });
	auto ctr_y = boxes.index({ Colon, 1 });
	auto widths = boxes.index({ Colon, 2 });
	auto heights = boxes.index({ Colon, 3 });
	auto angles = boxes.index({ Colon, 4 });

	auto dx = deltas.index({ Colon, 0 }) / m_weights.dx;
	auto dy = deltas.index({ Colon, 1 }) / m_weights.dy;
	auto dw = deltas.index({ Colon, 2 }) / m_weights.dw;
	auto dh = deltas.index({ Colon, 3 }) / m_weights.dh;
	auto da = deltas.index({ Colon, 4 }) / m_weights.da;

	// Prevent sending too large values into torch.exp()
	dw = torch::clamp(dw, nullopt, m_scale_clamp);
	dh = torch::clamp(dh, nullopt, m_scale_clamp);

	auto pred_boxes = torch::zeros_like(deltas);
	pred_boxes.index_put_({ Colon, 0 }, dx * widths + ctr_x);  // x_ctr
	pred_boxes.index_put_({ Colon, 1 }, dy * heights + ctr_y);  // y_ctr
	pred_boxes.index_put_({ Colon, 2 }, torch::exp(dw) * widths);  // width
	pred_boxes.index_put_({ Colon, 3 }, torch::exp(dh) * heights);  // height

	// Following original RRPN implementation,
	// angles of deltas are in radians while angles of boxes are in degrees.
	auto pred_angle = da * 180.0 / M_PI + angles;
	pred_angle = (pred_angle + 180.0) % 360.0 - 180.0;  // make it in [-180, 180)

	pred_boxes.index_put_({ Colon, 4 }, pred_angle);

	return pred_boxes;
}