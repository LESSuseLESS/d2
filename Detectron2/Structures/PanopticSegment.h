#pragma once

#include "Sequence.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	struct SegmentInfo {
		int id;
		bool isthing;
		float score;
		int category_id;
		int instance_id;
		float area;
	};

	class PanopticSegment : public Sequence {
	public:
		torch::Tensor seg;
		std::vector<SegmentInfo> infos;

		// implementing Sequence
		virtual int size() const override;
		virtual std::string toString() const override;
		virtual SequencePtr slice(int64_t start, int64_t end) const override;
		virtual SequencePtr index(torch::Tensor item) const override;
		virtual SequencePtr cat(const std::vector<SequencePtr> &seqs, int total) const override;
	};
}