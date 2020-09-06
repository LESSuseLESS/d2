#include "Base.h"
#include "RotatedBoxes.h"

#include <Detectron2/detectron2/box_iou_rotated/box_iou_rotated.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor RotatedBoxes::pairwise_iou_rotated(const torch::Tensor &boxes1, const torch::Tensor &boxes2) {
	return detectron2::box_iou_rotated(boxes1, boxes2);
}

torch::Tensor RotatedBoxes::pairwise_iou_rotated(const RotatedBoxes &boxes1, const RotatedBoxes &boxes2) {
	return pairwise_iou_rotated(boxes1.tensor(), boxes2.tensor());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RotatedBoxes::RotatedBoxes(const torch::Tensor &tensor) : Boxes() {
	m_tensor = tensor;
	if (m_tensor.numel() == 0) {
		// Use reshape, so we don't end up creating a new tensor that does not depend on
		// the inputs (and consequently confuses jit)
		m_tensor = m_tensor.reshape({ 0, 5 }).to(torch::kFloat32);
	}
	assert(m_tensor.dim() == 2 && m_tensor.size(-1) == 5);
}

RotatedBoxes::RotatedBoxes(const RotatedBoxes &boxes) : Boxes() {
	m_tensor = boxes.m_tensor;
}

torch::Tensor RotatedBoxes::area() const {
	return m_tensor.index({ Colon, 2 }) * m_tensor.index({ Colon, 3 });
}

void RotatedBoxes::normalize_angles() {
	m_tensor.index_put_({ Colon, 4 }, (m_tensor.index({ Colon, 4 }) + 180.0) % 360.0 - 180.0);
}

void RotatedBoxes::clip(BoxSizeType box_size, float clip_angle_threshold) {
	int h = box_size.height;
	int w = box_size.width;

	// normalize angles to be within (-180, 180] degrees
	normalize_angles();

	auto idx = torch::where(torch::abs(m_tensor.index({ Colon, 4 })) <= clip_angle_threshold)[0];

	// convert to (x1, y1, x2, y2)
	auto x1 = m_tensor[idx, 0] - m_tensor[idx, 2] / 2.0;
	auto y1 = m_tensor[idx, 1] - m_tensor[idx, 3] / 2.0;
	auto x2 = m_tensor[idx, 0] + m_tensor[idx, 2] / 2.0;
	auto y2 = m_tensor[idx, 1] + m_tensor[idx, 3] / 2.0;

	// clip
	x1.clamp_(0, w);
	y1.clamp_(0, h);
	x2.clamp_(0, w);
	y2.clamp_(0, h);

	// convert back to (xc, yc, w, h)
	m_tensor[idx, 0] = (x1 + x2) / 2.0;
	m_tensor[idx, 1] = (y1 + y2) / 2.0;
	// make sure widths and heights do not increase due to numerical errors
	m_tensor[idx, 2] = torch::min(m_tensor[idx, 2], x2 - x1);
	m_tensor[idx, 3] = torch::min(m_tensor[idx, 3], y2 - y1);
}

torch::Tensor RotatedBoxes::nonempty(float threshold) const {
	auto widths = m_tensor.index({ Colon, 2 });
	auto heights = m_tensor.index({ Colon, 3 });
	auto keep = (widths > threshold).bitwise_and(heights > threshold);
	return keep;
}

RotatedBoxes RotatedBoxes::operator[](int64_t item) const {
	return m_tensor[item].view({ 1, -1 });
}

RotatedBoxes RotatedBoxes::operator[](ArrayRef<torch::indexing::TensorIndex> item) const {
	auto b = m_tensor.index(item);
	assert(b.dim() == 2);
	return RotatedBoxes(b);
}

std::string RotatedBoxes::toString() const {
	return "RotatedBoxes(" + m_tensor.toString() + ")";
}

torch::Tensor RotatedBoxes::inside_box(BoxSizeType box_size, int boundary_threshold) const {
	auto cnt_x = m_tensor.index({ Ellipsis, 0 });
	auto cnt_y = m_tensor.index({ Ellipsis, 1 });
	auto half_w = m_tensor.index({ Ellipsis, 2 }) / 2.0;
	auto half_h = m_tensor.index({ Ellipsis, 3 }) / 2.0;
	auto a = m_tensor.index({ Ellipsis, 4 });
	auto c = torch::abs(torch::cos(a * M_PI / 180.0));
	auto s = torch::abs(torch::sin(a * M_PI / 180.0));
	// This basically computes the horizontal bounding rectangle of the rotated box
	auto max_rect_dx = c * half_w + s * half_h;
	auto max_rect_dy = c * half_h + s * half_w;

	auto inds_inside = ((cnt_x - max_rect_dx >= -boundary_threshold)
		.bitwise_and(cnt_y - max_rect_dy >= -boundary_threshold)
		.bitwise_and(cnt_x + max_rect_dx < box_size.width + boundary_threshold)
		.bitwise_and(cnt_y + max_rect_dy < box_size.height + boundary_threshold));

	return inds_inside;
}

torch::Tensor RotatedBoxes::get_centers() const {
	return m_tensor.index({ Colon, Slice(None, 2) });
}

void RotatedBoxes::scale(float scale_x, float scale_y) {
	m_tensor.index_put_({ Colon, 0 }, m_tensor * scale_x);
	m_tensor.index_put_({ Colon, 1 }, m_tensor * scale_y);
	auto theta = m_tensor.index({ Colon, 4 }) * M_PI / 180.0;
	auto c = torch::cos(theta);
	auto s = torch::sin(theta);

    // In image space, y is top->down and x is left->right
    // Consider the local coordintate system for the rotated box,
    // where the box center is located at (0, 0), and the four vertices ABCD are
    // A(-w / 2, -h / 2), B(w / 2, -h / 2), C(w / 2, h / 2), D(-w / 2, h / 2)
    // the midpoint of the left edge AD of the rotated box E is:
    // E = (A+D)/2 = (-w / 2, 0)
    // the midpoint of the top edge AB of the rotated box F is:
    // F(0, -h / 2)
    // To get the old coordinates in the global system, apply the rotation transformation
    // (Note: the right-handed coordinate system for image space is yOx):
    // (old_x, old_y) = (s * y + c * x, c * y - s * x)
    // E(old) = (s * 0 + c * (-w/2), c * 0 - s * (-w/2)) = (-c * w / 2, s * w / 2)
    // F(old) = (s * (-h / 2) + c * 0, c * (-h / 2) - s * 0) = (-s * h / 2, -c * h / 2)
    // After applying the scaling factor (sfx, sfy):
    // E(new) = (-sfx * c * w / 2, sfy * s * w / 2)
    // F(new) = (-sfx * s * h / 2, -sfy * c * h / 2)
    // The new width after scaling tranformation becomes:

    // w(new) = |E(new) - O| * 2
    //        = sqrt[(sfx * c * w / 2)^2 + (sfy * s * w / 2)^2] * 2
    //        = sqrt[(sfx * c)^2 + (sfy * s)^2] * w
    // i.e., scale_factor_w = sqrt[(sfx * c)^2 + (sfy * s)^2]
    //
    // For example,
    // when angle = 0 or 180, |c| = 1, s = 0, scale_factor_w == scale_factor_x;
    // when |angle| = 90, c = 0, |s| = 1, scale_factor_w == scale_factor_y
	m_tensor.index_put_({ Colon, 2 },
		m_tensor.index({ Colon, 2 }) * torch::sqrt((scale_x * c).pow(2) + (scale_y * s).pow(2)));

    // h(new) = |F(new) - O| * 2
    //        = sqrt[(sfx * s * h / 2)^2 + (sfy * c * h / 2)^2] * 2
    //        = sqrt[(sfx * s)^2 + (sfy * c)^2] * h
    // i.e., scale_factor_h = sqrt[(sfx * s)^2 + (sfy * c)^2]
    //
    // For example,
    // when angle = 0 or 180, |c| = 1, s = 0, scale_factor_h == scale_factor_y;
    // when |angle| = 90, c = 0, |s| = 1, scale_factor_h == scale_factor_x
	m_tensor.index_put_({ Colon, 3 },
		m_tensor.index({ Colon, 3 }) * torch::sqrt((scale_x * s).pow(2) + (scale_y * c).pow(2)));

    // The angle is the rotation angle from y-axis in image space to the height
    // vector (top->down in the box's local coordinate system) of the box in CCW.
    //
    // angle(new) = angle_yOx(O - F(new))
    //            = angle_yOx( (sfx * s * h / 2, sfy * c * h / 2) )
    //            = atan2(sfx * s * h / 2, sfy * c * h / 2)
    //            = atan2(sfx * s, sfy * c)
    //
    // For example,
    // when sfx == sfy, angle(new) == atan2(s, c) == angle(old)
	m_tensor.index_put_({ Colon, 4 }, torch::atan2(scale_x * s, scale_y * c) * 180 / M_PI);
}