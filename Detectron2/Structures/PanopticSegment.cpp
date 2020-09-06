#include "Base.h"
#include "PanopticSegment.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int PanopticSegment::size() const {
	return infos.size();
}

std::string PanopticSegment::toString() const {
	// TODO: infos
	return seg.toString();
}

SequencePtr PanopticSegment::slice(int64_t start, int64_t end) const {
	assert(false);
	return nullptr;
}

SequencePtr PanopticSegment::index(torch::Tensor item) const {
	assert(false);
	return nullptr;
}

SequencePtr PanopticSegment::cat(const std::vector<SequencePtr> &seqs, int total) const {
	assert(false);
	return nullptr;
}
