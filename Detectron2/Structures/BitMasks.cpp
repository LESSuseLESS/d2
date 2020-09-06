#include "Base.h"
#include "BitMasks.h"

#include <Detectron2/Modules/ROIPooler/ROIAlign.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BitMasks::BitMasks(const torch::Tensor &tensor) {
	m_tensor = tensor.to(torch::kBool);
	assert(m_tensor.dim() == 3);
	m_image_size = { (int)m_tensor.size(1), (int)m_tensor.size(2) };
}

BitMasks::BitMasks(const BitMasks &bitmasks) : m_tensor(bitmasks.m_tensor), m_image_size(bitmasks.m_image_size) {
}

BitMasks BitMasks::operator[](int64_t item) const {
	return m_tensor[item].view({ 1, -1 });
}

BitMasks BitMasks::operator[](ArrayRef<torch::indexing::TensorIndex> item) const {
	auto m = m_tensor.index(item);
	assert(m.dim() == 3);
	return m;
}

std::string BitMasks::toString() const {
	string s = "BitMasks(";
	s += FormatString("num_instances=%d)", size());
	return s;
}

SequencePtr BitMasks::slice(int64_t start, int64_t end) const {
	auto sliced = m_tensor.slice(0, start, end);
	return std::shared_ptr<BitMasks>(new BitMasks(sliced));
}

SequencePtr BitMasks::index(torch::Tensor item) const {
	auto selected = m_tensor.index(item);
	return std::shared_ptr<BitMasks>(new BitMasks(selected));
}

SequencePtr BitMasks::cat(const std::vector<SequencePtr> &seqs, int total) const {
	TensorVec tensors;
	tensors.reserve(seqs.size());
	for (auto &seq : seqs) {
		Tensor t = dynamic_pointer_cast<BitMasks>(seq)->m_tensor;
		tensors.push_back(t);
	}
	auto aggregated = torch::cat(tensors);
	assert(aggregated.size(0) == total);
	return std::shared_ptr<BitMasks>(new BitMasks(aggregated));
}

torch::Tensor BitMasks::crop_and_resize(torch::Tensor boxes, int mask_size) {
	assert(boxes.size(0) == size());
	auto device = m_tensor.device();

	auto batch_inds = torch::arange(size(), device).to(boxes.dtype()).index({ Colon, None });
	auto rois = torch::cat({ batch_inds, boxes }, 1);  // Nx5

	auto bit_masks = m_tensor.to(torch::kFloat32);
	rois = rois.to(device);
	auto output = (
		ROIAlignImpl({ mask_size, mask_size }, 1.0, 0, true)
		.forward(bit_masks.index({ Colon, None, Colon, Colon }), rois)
		.squeeze(1)
		);
	output = (output >= 0.5);
	return output;
}
