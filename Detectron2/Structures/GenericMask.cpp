#include "Base.h"
#include "GenericMask.h"

#include <Detectron2/Utils/Utils.h>

using namespace std;
using namespace cv;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor GenericMask::toCocoMask(const std::vector<std::shared_ptr<GenericMask>> &masks) {
	TensorVec ret;
	ret.reserve(masks.size());
	for (auto mask : masks) {
		ret.push_back(mask->mask());
	}
	return torch::stack(ret);
}

std::vector<std::shared_ptr<GenericMask>> GenericMask::_convert_masks(const BitMasks &m, int height, int width) {
	std::vector<std::shared_ptr<GenericMask>> ret;
	auto t = m.tensor();
	int count = t.size(0);
	ret.reserve(count);
	for (int i = 0; i < count; i++) {
		ret.push_back(make_shared<GenericMask>(t[i], height, width));
	}
	return ret;
}

std::vector<std::shared_ptr<GenericMask>> GenericMask::_convert_masks(const PolygonMasks &m, int height, int width) {
	std::vector<std::shared_ptr<GenericMask>> ret;
	auto t = m.polygons();
	int count = t.size();
	ret.reserve(count);
	for (int i = 0; i < count; i++) {
		ret.push_back(make_shared<GenericMask>(t[i], height, width));
	}
	return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GenericMask::GenericMask(const torch::Tensor &mask, int height, int width) :
	m_height(height), m_width(width), m_has_mask(true), m_has_polygons(false), m_has_holes(0)
{
	assert(mask.size(1) != 2);
	assert(mask.size(0) == height);
	assert(mask.size(1) == width);
	m_mask = mask.to(torch::kUInt8);
}

GenericMask::GenericMask(const TensorVec &polygons, int height, int width) :
	m_height(height), m_width(width), m_has_mask(false), m_has_polygons(true), m_has_holes(0)
{
	int count = polygons.size();
	m_polygons.reserve(count);
	for (int i = 0; i < count; i++) {
		m_polygons.push_back(polygons[i].reshape(-1));
	}
}

GenericMask::GenericMask(const mask_util::MaskObject &obj, int height, int width) {
	if (!obj->counts_uncompressed.empty()) {
		auto h = obj->size.height;
		auto w = obj->size.width;
		assert(h == height && w == width);
		m_mask = mask_util::decode_single(mask_util::frPyObjects_single(obj, h, w));
	}
	else {
		assert(!obj->counts.empty());
		m_mask = mask_util::decode_single(obj).index({ Colon, Colon });
	}
}

torch::Tensor GenericMask::mask() {
	if (!m_has_mask) {
		m_mask = polygons_to_mask(m_polygons);
		m_has_mask = true;
	}
	return m_mask;
}

TensorVec GenericMask::polygons() {
	if (!m_has_polygons) {
		tie(m_polygons, m_has_holes) = mask_to_polygons(m_mask);
		m_has_polygons = true;
	}
	return m_polygons;
}

bool GenericMask::has_holes() {
	if (m_has_holes == 0) {
		if (m_has_mask) {
			tie(m_polygons, m_has_holes) = mask_to_polygons(m_mask);
			m_has_polygons = true;
		}
		else {
			m_has_holes = -1; // if original format is polygon, does not have holes
		}
	}
	return m_has_holes > 0;
}

std::tuple<TensorVec, int> GenericMask::mask_to_polygons(const torch::Tensor &mask) {
	// cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
	// hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
	// Internal contours (holes) are placed in hierarchy-2.
	// cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.

	cv::Mat mat_mask = image_to_mat(mask.to(torch::kUInt8)); // some versions of cv2 does not support incontiguous arr
	vector<vector<cv::Point>> contours;
	vector<Vec4i> vvhierarchy;
	cv::findContours(mat_mask, contours, vvhierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
	if (contours.empty()) { // empty mask
		return { TensorVec{}, -1 };
	}
	TensorVec vres;
	vres.reserve(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		vector<int64_t> pts;
		auto &vpts = contours[i];
		auto count = vpts.size();
		if (count >= 6) {
			pts.reserve(count * 2);
			for (int j = 0; j < count; j++) {
				pts.push_back(vpts[j].x);
				pts.push_back(vpts[j].y);
			}
			Tensor temp = torch::tensor(pts, dtype(torch::kInt64)).reshape({ -1, 2 });
			vres.push_back(temp);
		}
	}

	TensorVec vhierarchy;
	vhierarchy.reserve(vvhierarchy.size());
	for (int i = 0; i < vvhierarchy.size(); i++) {
		auto &vpts = vvhierarchy[i];
		vector<int64_t> pts{ vpts[0], vpts[1], vpts[2], vpts[3] };
		vhierarchy.push_back(torch::tensor(pts));
	}
	auto hierarchy = torch::cat(vhierarchy);
	bool has_holes = ((hierarchy.reshape({ -1, 4 }).index({ Colon, 3 }) >= 0).sum().item<int64_t>() > 0);

	return { vres, has_holes ? 1 : -1 };
}

torch::Tensor GenericMask::polygons_to_mask(const TensorVec &polygons) {
	auto rle = mask_util::frPyObjects_polygons(polygons, m_height, m_width);
	auto merged = mask_util::merge(rle);
	return mask_util::decode_single(merged).index({ Colon, Colon });
}

torch::Tensor GenericMask::bbox() const {
	assert(m_has_polygons);
	auto p = mask_util::frPyObjects_polygons(m_polygons, m_height, m_width);
	auto merged = mask_util::merge(p);
	auto bbox = mask_util::toBbox_single(merged);
	bbox.index_put_({ 2 }, bbox[2] + bbox[0]);
	bbox.index_put_({ 3 }, bbox[3] + bbox[1]);
	return bbox;
}
