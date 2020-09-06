#include "Base.h"
#include "Matcher.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Matcher::Matcher(const vector<float> &thresholds, const vector<int> &labels, bool allow_low_quality_matches) :
	m_thresholds(thresholds), m_labels(labels), m_allow_low_quality_matches(allow_low_quality_matches) {
	assert(m_thresholds[0] > 0);
	m_thresholds.insert(m_thresholds.begin(), -INFINITY);
	m_thresholds.push_back(INFINITY);
	for (int i = 0; i < m_thresholds.size() - 1; i++) {
		assert(m_thresholds[i] < m_thresholds[i + 1]);
	}
	for (auto label : m_labels) {
		assert(label == 0 || abs(label) == 1);
	}
	assert(m_labels.size() == m_thresholds.size() - 1);
}

std::tuple<torch::Tensor, torch::Tensor> Matcher::operator()(const torch::Tensor &match_quality_matrix) {
	assert(match_quality_matrix.dim() == 2);
	int count = match_quality_matrix.size(0);

	if (count == 0) {
		auto default_matches = match_quality_matrix.new_full(
			{ match_quality_matrix.size(1) }, 0, torch::kInt64
		);
		// When no gt boxes exist, we define IOU = 0 and therefore set labels
		// to `self.labels[0]`, which usually defaults to background class 0
		// To choose to ignore instead, can make labels=[-1,0,-1,1] + set appropriate thresholds
		auto default_match_labels = match_quality_matrix.new_full(
			{ match_quality_matrix.size(1) }, m_labels[0], torch::kInt8
		);
		return { default_matches, default_match_labels };
	}

	assert(torch::all(match_quality_matrix >= 0).item<bool>());

	// match_quality_matrix is M (gt) x N (predicted)
	// Max over gt elements (dim 0) to find best gt candidate for each prediction
	torch::Tensor matched_vals, matches;
	tie(matched_vals, matches) = match_quality_matrix.max(0);

	auto match_labels = matches.new_full(matches.sizes(), 1, torch::kInt8);
	for (int i = 0; i < m_labels.size(); i++) {
		auto label = m_labels[i];
		auto low = m_thresholds[i];
		auto high = m_thresholds[i + 1];
		auto low_high = (matched_vals >= low).bitwise_and(matched_vals < high);
		match_labels.index_put_({ low_high }, label);
	}

	if (m_allow_low_quality_matches) {
		set_low_quality_matches_(match_quality_matrix, match_labels);
	}
	return { matches, match_labels };
}

void Matcher::set_low_quality_matches_(const torch::Tensor &match_quality_matrix, torch::Tensor &match_labels) {
	// For each gt, find the prediction with which it has highest quality
	auto highest_quality_foreach_gt = match_quality_matrix.max_values(1);
	// Find the highest quality match available, even if it is low, including ties.
    // Note that the matches qualities must be positive due to the use of
    // `torch.nonzero`.
	auto pred_inds_with_highest_quality = torch::nonzero(
		match_quality_matrix == highest_quality_foreach_gt.index({ Colon, None })
	).index({ Colon, -1 });
    // If an anchor was labeled positive only due to a low-quality match
    // with gt_A, but it has larger overlap with gt_B, it's matched index will still be gt_B.
    // This follows the implementation in Detectron, and is found to have no significant impact.
	match_labels.index_put_({ pred_inds_with_highest_quality }, 1);
}
