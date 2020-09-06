#include "Base.h"
#include "Boxes.h"
#include "RotatedBoxes.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// not converted

/*
_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]
@unique
class BoxMode(IntEnum):
	"""
	Enum of different ways to represent a box.
	"""

	XYXY_ABS = 0
	"""
	(x0, y0, x1, y1) in absolute floating points coordinates.
	The coordinates in range [0, width or height].
	"""
	XYWH_ABS = 1
	"""
	(x0, y0, w, h) in absolute floating points coordinates.
	"""
	XYXY_REL = 2
	"""
	Not yet supported!
	(x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
	"""
	XYWH_REL = 3
	"""
	Not yet supported!
	(x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
	"""
	XYWHA_ABS = 4
	"""
	(xc, yc, w, h, a) in absolute floating points coordinates.
	(xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
	"""

	@staticmethod
	def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
		"""
		Args:
			box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
			from_mode, to_mode (BoxMode)

		Returns:
			The converted box of the same type.
		"""
		if from_mode == to_mode:
			return box

		original_type = type(box)
		is_numpy = isinstance(box, np.ndarray)
		single_box = isinstance(box, (list, tuple))
		if single_box:
			assert len(box) == 4 or len(box) == 5, (
				"BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
				" where k == 4 or 5"
			)
			arr = torch.tensor(box)[None, :]
		else:
			// avoid modifying the input box
			if is_numpy:
				arr = torch.from_numpy(np.asarray(box)).clone()
			else:
				arr = box.clone()

		assert to_mode.value not in [
			BoxMode.XYXY_REL,
			BoxMode.XYWH_REL,
		] and from_mode.value not in [
			BoxMode.XYXY_REL,
			BoxMode.XYWH_REL,
		], "Relative mode not yet supported!"

		if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
			assert (
				arr.shape[-1] == 5
			), "The last dimension of input shape must be 5 for XYWHA format"
			original_dtype = arr.dtype
			arr = arr.double()

			w = arr[:, 2]
			h = arr[:, 3]
			a = arr[:, 4]
			c = torch.abs(torch.cos(a * math.pi / 180.0))
			s = torch.abs(torch.sin(a * math.pi / 180.0))
			// This basically computes the horizontal bounding rectangle of the rotated box
			new_w = c * w + s * h
			new_h = c * h + s * w

			// convert center to top-left corner
			arr[:, 0] -= new_w / 2.0
			arr[:, 1] -= new_h / 2.0
			// bottom-right corner
			arr[:, 2] = arr[:, 0] + new_w
			arr[:, 3] = arr[:, 1] + new_h

			arr = arr[:, :4].to(dtype=original_dtype)
		elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
			original_dtype = arr.dtype
			arr = arr.double()
			arr[:, 0] += arr[:, 2] / 2.0
			arr[:, 1] += arr[:, 3] / 2.0
			angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
			arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
		else:
			if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
				arr[:, 2] += arr[:, 0]
				arr[:, 3] += arr[:, 1]
			elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
				arr[:, 2] -= arr[:, 0]
				arr[:, 3] -= arr[:, 1]
			else:
				raise NotImplementedError(
					"Conversion from BoxMode {} to {} is not supported yet".format(
						from_mode, to_mode
					)
				)

		if single_box:
			return original_type(arr.flatten().tolist())
		if is_numpy:
			return arr.numpy()
		else:
			return arr
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor Boxes::pairwise_iou(const Boxes &boxes1_, const Boxes &boxes2_) {
	auto area1 = boxes1_.area();
	auto area2 = boxes2_.area();

	auto boxes1 = boxes1_.tensor();
	auto boxes2 = boxes2_.tensor();

	auto width_height = torch::min(
		boxes1.index({ Colon, None, Slice(2, None) }),
		boxes2.index({ Colon, Slice(2, None) })) - torch::max(
			boxes1.index({ Colon, None, Slice(None, 2) }),
			boxes2.index({ Colon, Slice(None, 2) })); // [N,M,2]

	width_height.clamp_(0);  // [N,M,2]
	auto inter = width_height.prod(2);  // [N,M]

	// handle empty boxes
	auto iou = torch::where(
			inter > 0,
			inter / (area1.index({ Colon, None }) + area2 - inter),
			torch::zeros(1, dtype(inter.dtype()).device(inter.device()))
		);
	return iou;
}

torch::Tensor matched_boxlist_iou(const Boxes &boxes1, const Boxes &boxes2) {
	assert(boxes1.len() == boxes2.len()); // boxlists should have the same" "number of entries
	auto area1 = boxes1.area();  // [N]
	auto area2 = boxes2.area();  // [N]
	auto box1 = boxes1.tensor();
	auto box2 = boxes2.tensor();
	auto lt = torch::max(box1.index({ Colon, Slice(None, 2) }), box2.index({ Colon, Slice(None, 2) }));  // [N,2]
	auto rb = torch::min(box1.index({ Colon, Slice(2, None) }), box2.index({ Colon, Slice(2, None) }));  // [N,2]
	auto wh = (rb - lt).clamp(0);  // [N,2]
	auto inter = wh.index({ Colon, 0 }) * wh.index({ Colon, 1 });  // [N]
	auto iou = inter / (area1 + area2 - inter);  // [N]
	return iou;
}

torch::Tensor Boxes::cat(const BoxesList &boxes_list) {
	int count = boxes_list.size();
	if (count == 0) {
		return torch::empty(0);
	}
	return torch::cat(boxes_list);
}

std::shared_ptr<Boxes> Boxes::boxes(const torch::Tensor &tensor) {
	switch (tensor.size(-1)) {
	case 4:
		return make_shared<Boxes>(tensor);
	case 5:
		return make_shared<RotatedBoxes>(tensor);
	}
	assert(false);
	return nullptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Boxes::Boxes(const Tensor &tensor) : m_tensor(tensor) {
	if (m_tensor.numel() == 0) {
		// Use reshape, so we don't end up creating a new tensor that does not depend on
		// the inputs (and consequently confuses jit)
		m_tensor = m_tensor.reshape({ 0, 4 }).to(torch::kFloat32);
	}
	assert(m_tensor.dim() == 2 && m_tensor.size(-1) == 4);
}

Boxes::Boxes(const Boxes &boxes) : m_tensor(boxes.m_tensor) {
}

void Boxes::clip(BoxSizeType box_size, float) {
	assert(torch::isfinite(m_tensor).all().item<bool>());
	int h = box_size.height;
	int w = box_size.width;
	lefts().clamp_(0, w); rights().clamp_(0, w);
	tops().clamp_(0, h); bottoms().clamp_(0, h);
}

Tensor Boxes::nonempty(float threshold) const {
	return (widths() > threshold).logical_and(heights() > threshold);
}

Boxes Boxes::operator[](int64_t item) const {
	return m_tensor[item].view({ 1, -1 });
}

Boxes Boxes::operator[](ArrayRef<torch::indexing::TensorIndex> item) const {
	auto b = m_tensor.index(item);
	assert(b.dim() == 2);
	return b;
}

std::string Boxes::toString() const {
	return "Boxes(" + m_tensor.toString() + ")";
}

std::tuple<int, int, int, int> Boxes::bbox(int index) const {
	auto b = tolist(m_tensor[index]);
	return {
		(int)b[0].item<float>(),
		(int)b[1].item<float>(),
		(int)b[2].item<float>(),
		(int)b[3].item<float>()
	};
}

Tensor Boxes::inside_box(BoxSizeType box_size, int boundary_threshold) const {
	return (lefts() >= -boundary_threshold)
		.logical_and(tops() >= -boundary_threshold)
		.logical_and(rights() < (box_size.width + boundary_threshold))
		.logical_and(bottoms() < (box_size.height + boundary_threshold));
}

Tensor Boxes::get_centers() const {
	return (m_tensor.index({ Colon, Slice(None, 2) }) + m_tensor.index({ Colon, Slice(2, None) })) / 2;
}

void Boxes::scale(float scale_x, float scale_y) {
	m_tensor.index_put_({ Colon, Slice(0, None, 2) }, m_tensor.index({ Colon, Slice(0, None, 2) }) * scale_x);
	m_tensor.index_put_({ Colon, Slice(1, None, 2) }, m_tensor.index({ Colon, Slice(1, None, 2) }) * scale_y);
}
