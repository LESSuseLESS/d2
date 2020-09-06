#include "Base.h"
#include "Sampling.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor>
Detectron2::subsample_labels(const torch::Tensor &labels, int num_samples, float positive_fraction, int bg_label) {
	auto positive = torch::nonzero((labels != -1).bitwise_and(labels != bg_label)).index({ Colon, 0 });
	auto negative = torch::nonzero(labels == bg_label).index({ Colon, 0 });

	auto num_pos = int64_t(num_samples * positive_fraction);
	// protect against not enough positive examples
	num_pos = min(positive.numel(), num_pos);
	auto num_neg = num_samples - num_pos;
	// protect against not enough negative examples
	num_neg = min(negative.numel(), num_neg);

	// randomly select positive and negative examples
	auto perm1 = torch::randperm(positive.numel(), positive.device()).index({ Slice(None, num_pos) });
	auto perm2 = torch::randperm(negative.numel(), negative.device()).index({ Slice(None, num_neg) });

	auto pos_idx = positive.index(perm1);
	auto neg_idx = negative.index(perm2);
	return { pos_idx, neg_idx };
}
