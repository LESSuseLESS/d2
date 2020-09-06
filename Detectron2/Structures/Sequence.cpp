#include "Base.h"
#include "Sequence.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

SequencePtr SequenceTensor::slice(int64_t start, int64_t end) const {
	auto sliced = m_data.slice(0, start, end);
	return std::shared_ptr<SequenceTensor>(new SequenceTensor(sliced));
}

SequencePtr SequenceTensor::index(torch::Tensor item) const {
	auto selected = m_data.index(item);
	return std::shared_ptr<SequenceTensor>(new SequenceTensor(selected));
}

SequencePtr SequenceTensor::cat(const std::vector<SequencePtr> &seqs, int total) const {
	TensorVec tensors;
	tensors.reserve(seqs.size());
	for (auto &seq : seqs) {
		Tensor t = dynamic_pointer_cast<SequenceTensor>(seq)->m_data;
		tensors.push_back(t);
	}
	auto aggregated = torch::cat(tensors);
	assert(aggregated.size(0) == total);
	return std::shared_ptr<SequenceTensor>(new SequenceTensor(aggregated));
}
