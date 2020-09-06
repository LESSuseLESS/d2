#include "Base.h"
#include "PolygonMasks.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PolygonMasks PolygonMasks::cat(const PolygonMasksList &polymasks_list) {
	assert(!polymasks_list.empty());

	PolygonMasks ret;
	auto &cat_polymasks = ret.m_polygons;
	int count = 0;
	for (int i = 0; i < polymasks_list.size(); i++) {
		count += polymasks_list[i]->m_polygons.size();
	}
	cat_polymasks.reserve(count);
	for (int i = 0; i < polymasks_list.size(); i++) {
		auto &p = polymasks_list[i]->m_polygons;
		cat_polymasks.insert(cat_polymasks.end(), p.begin(), p.end());
	}
    return ret;
}

float PolygonMasks::polygon_area(const torch::Tensor &x, const torch::Tensor &y) {
	// Using the shoelace formula
	// https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
	return (0.5 * abs(dot(x, roll(y, 1)) - dot(y, roll(x, 1)))).item<float>();
}

torch::Tensor PolygonMasks::polygons_to_bitmask(const TensorVec &polygons, int height, int width) {
	assert(!polygons.empty()); // "COCOAPI does not support empty polygons"
	auto rles = mask_util::frPyObjects_polygons(polygons, height, width);
	auto merged = mask_util::merge(rles);
	return mask_util::decode_single(merged).to(torch::kBool);
}

torch::Tensor PolygonMasks::rasterize_polygons_within_box(const TensorVec &polygons_, const torch::Tensor &box,
	int mask_size) {
	// 1. Shift the polygons w.r.t the boxes
	auto w = (box[2] - box[0]).item<float>();
	auto h = (box[3] - box[1]).item<float>();

	TensorVec polygons;
	polygons.reserve(polygons_.size());
	for (int i = 0; i < polygons_.size(); i++) {
		polygons.push_back(polygons_[i].clone());
	}
	for (int i = 0; i < polygons.size(); i++) {
		auto &p = polygons[i];
		p.index_put_({ Slice(0, None, 2) }, p.index({ Slice(0, None, 2) }) - box[0]);
		p.index_put_({ Slice(1, None, 2) }, p.index({ Slice(1, None, 2) }) - box[1]);
	}

	// 2. Rescale the polygons to the new box size
	// max() to avoid division by small number
	auto ratio_h = mask_size / max(h, 0.1f);
	auto ratio_w = mask_size / max(w, 0.1f);

	if (ratio_h == ratio_w) {
		for (int i = 0; i < polygons.size(); i++) {
			auto &p = polygons[i];
			p *= ratio_h;
		}
	}
	else {
		for (int i = 0; i < polygons.size(); i++) {
			auto &p = polygons[i];
			p.index_put_({ Slice(0, None, 2) }, p.index({ Slice(0, None, 2) }) * ratio_w);
			p.index_put_({ Slice(1, None, 2) }, p.index({ Slice(1, None, 2) }) * ratio_h);
		}
	}

	// 3. Rasterize the polygons with coco api
	auto mask = polygons_to_bitmask(polygons, mask_size, mask_size);
	return mask;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PolygonMasks::PolygonMasks(std::vector<TensorVec> polygons) : m_polygons(std::move(polygons)) {
	for (int i = 0; i < m_polygons.size(); i++) {
		auto &polygons_per_instance = m_polygons[i];
		for (int j = 0; j < polygons_per_instance.size(); j++) {
			auto &p = polygons_per_instance[j];

			// Use float64 for higher precision, because why not?
			// Always put polygons on CPU (self.to is a no-op) since they
			// are supposed to be small tensors.
			// May need to change this assumption if GPU placement becomes useful
			p = p.to(kFloat64);

			int len = p.size(0);
			assert(len % 2 == 0 && len >= 6);
		}
	}
}

BitMasks PolygonMasks::from_polygon_masks(int height, int width) const {
	TensorVec masks;
	masks.reserve(m_polygons.size());
	for (int i = 0; i < m_polygons.size(); i++) {
		auto &p = m_polygons[i];
		masks.push_back(polygons_to_bitmask(p, height, width));
	}
	return torch::stack(masks);
}

Boxes PolygonMasks::get_bounding_boxes() const {
	auto boxes = torch::zeros({ size(), 4 }, torch::kFloat32);
	for (int idx = 0; idx < m_polygons.size(); idx++) {
		auto &polygons_per_instance = m_polygons[idx];
		auto minxy = torch::tensor({ numeric_limits<float>::max(), numeric_limits<float>::max() }, torch::kFloat32);
		auto maxxy = torch::zeros(2, torch::kFloat32);
		for (int j = 0; j < polygons_per_instance.size(); j++) {
			auto polygon = polygons_per_instance[j];
			auto coords = polygon.view({ -1, 2 }).to(torch::kFloat32);
			minxy = torch::min(minxy, torch::min_values(coords, 0));
			maxxy = torch::max(maxxy, torch::max_values(coords, 0));
		}
		boxes.index_put_({ idx, Slice(None, 2) }, minxy);
		boxes.index_put_({ idx, Slice(2, None) }, maxxy);
	}
	return boxes;
}

torch::Tensor PolygonMasks::nonempty() {
	vector<int64_t> keep;
	keep.reserve(m_polygons.size());
	for (int i = 0; i < m_polygons.size(); i++) {
		auto &polygon = m_polygons[i];
		keep.push_back(polygon.size() ? 0 : 1);
	}
	return torch::tensor(keep).to(torch::kBool);
}

std::string PolygonMasks::toString() const {
	string s = "PolygonMasks(";
	s += FormatString("num_instances=%d)", size());
	return s;
}

PolygonMasks PolygonMasks::operator[](int64_t item) const {
	PolygonMasks ret;
	ret.m_polygons = { m_polygons[item] };
	return ret;
}

PolygonMasks PolygonMasks::operator[](const std::vector<int64_t> &item) const {
	PolygonMasks ret;
	auto &polygons = ret.m_polygons;
	polygons.reserve(item.size());
	for (int i = 0; i < item.size(); i++) {
		polygons.push_back(m_polygons[item[i]]);
	}
	return ret;
}

/*~!
PolygonMasks PolygonMasks::operator[](torch::ArrayRef<torch::indexing::TensorIndex> item) const {
	elif isinstance(item, slice):
		selected_polygons = self.polygons[item]
	elif isinstance(item, torch.Tensor):
		// Polygons is a list, so we have to move the indices back to CPU.
		if item.dtype == torch.bool:
			assert item.dim() == 1, item.shape
			item = item.nonzero().squeeze(1).cpu().numpy().tolist()
		elif item.dtype in [torch.int32, torch.int64]:
			item = item.cpu().numpy().tolist()
		else:
			raise ValueError("Unsupported tensor dtype={} for indexing!".format(item.dtype))
		selected_polygons = [self.polygons[i] for i in item]
	return PolygonMasks(selected_polygons)
}
*/

SequencePtr PolygonMasks::slice(int64_t start, int64_t end) const {
	assert(start >= 0 && start < m_polygons.size());
	assert(end >= 0 && end <= m_polygons.size());

	int count = (end - start - 1);
	std::vector<TensorVec> sliced;
	if (count > 0) {
		sliced.reserve(count);
		for (int i = start; i < end; i++) {
			sliced.push_back(m_polygons[i]);
		}
	}
	return shared_ptr<PolygonMasks>(new PolygonMasks(std::move(sliced)));
}

SequencePtr PolygonMasks::index(torch::Tensor item) const {
	assert(item.dim() == 1 && item.dtype() == torch::kInt64);
	std::vector<TensorVec> selected;
	int count = item.size(0);
	selected.reserve(count);
	for (int i = 0; i < count; i++) {
		auto index = item[i].item<int64_t>();
		assert(index >= 0 && index < m_polygons.size());
		selected.push_back(m_polygons[i]);
	}
	return shared_ptr<PolygonMasks>(new PolygonMasks(std::move(selected)));
}

SequencePtr PolygonMasks::cat(const std::vector<SequencePtr> &seqs, int total) const {
	std::vector<TensorVec> aggregated;
	aggregated.reserve(total);
	for (int i = 0; i < seqs.size(); i++) {
		auto &p = std::dynamic_pointer_cast<PolygonMasks>(seqs[i])->m_polygons;
		aggregated.insert(aggregated.end(), p.begin(), p.end());
	}
	assert(aggregated.size() == total);
	return shared_ptr<PolygonMasks>(new PolygonMasks(std::move(aggregated)));
}

torch::Tensor PolygonMasks::crop_and_resize(torch::Tensor boxes, int mask_size) {
	assert(boxes.size(0) == size());

	auto device = boxes.device();
	// Put boxes on the CPU, as the polygon representation is not efficient GPU-wise
	// (several small tensors for representing a single instance mask)
	boxes = boxes.cpu();

	TensorVec results;
	results.reserve(size());
	for (int i = 0; i < m_polygons.size(); i++) {
		auto &poly = m_polygons[i];
		auto box = boxes[i];
		results.push_back(rasterize_polygons_within_box(poly, box, mask_size));
	}

	// poly: list[list[float]], the polygons for one instance
	// box: a tensor of shape (4,)
	if (results.empty()) {
		return torch::empty({ 0, mask_size, mask_size }, dtype(torch::kBool).device(device));
	}
	return torch::stack(results, 0).to(device);
}

torch::Tensor PolygonMasks::area() {
	int count = size();
	vector<float> area;
	area.reserve(count);
	for (int i = 0; i < count; i++) {
		auto &polygons_per_instance = m_polygons[i];
		float area_per_instance = 0;
		for (int j = 0; j < polygons_per_instance.size(); j++) {
			auto p = polygons_per_instance[j];
			area_per_instance += polygon_area(p.index({ Slice(0, None, 2) }), p.index({ Slice(1, None, 2) }));
		}
		area.push_back(area_per_instance);
	}
	return torch::tensor(area);
}
