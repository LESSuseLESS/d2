#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/matcher.py

	/**
		This class assigns to each predicted "element" (e.g., a box) a ground-truth element. Each predicted element
		will have exactly zero or one matches; each ground-truth element may be matched to zero or more predicted
		elements.

		The matching is determined by the MxN match_quality_matrix, that characterizes how well each (ground-truth,
		prediction)-pair match each other. For example, if the elements are boxes, this matrix may contain box
		intersection-over-union overlap values.

		The matcher returns (a) a vector of length N containing the index of the ground-truth element m in [0, M) that
		matches to prediction n in [0, N). (b) a vector of length N containing the labels for each prediction.
	*/
	class Matcher {
	public:
		/**
			thresholds: a list of thresholds used to stratify predictions into levels.
			labels: a list of values to label predictions belonging at each level. A label can be one of {-1, 0, 1}
				signifying {ignore, negative class, positive class}, respectively.
			allow_low_quality_matches: if True, produce additional matches for predictions with maximum match quality
				lower than high_threshold. See set_low_quality_matches_ for more details.

			For example,
				thresholds = [0.3, 0.5]
				labels = [0, -1, 1]
				All predictions with iou < 0.3 will be marked with 0 and thus will be considered as false positives
					while training.
				All predictions with 0.3 <= iou < 0.5 will be marked with -1 and thus will be ignored.
				All predictions with 0.5 <= iou will be marked with 1 and thus will be considered as true positives.
		*/
		Matcher(const std::vector<float> &thresholds, const std::vector<int> &labels,
			bool allow_low_quality_matches = false);

		/**
			match_quality_matrix (Tensor[float]): an MxN tensor, containing the pairwise quality between M ground-truth
				elements and N predicted elements. All elements must be >= 0 (due to the us of `torch.nonzero` for
				selecting indices in :meth:`set_low_quality_matches_`).

			Returns:
				matches (Tensor[int64]): a vector of length N, where matches[i] is a matched ground-truth index
					in [0, M)
				match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates whether a
					prediction is a true or false positive or ignored
		*/
		std::tuple<torch::Tensor, torch::Tensor> operator()(const torch::Tensor &match_quality_matrix);

		/**
			Produce additional matches for predictions that have only low-quality matches. Specifically, for each
			ground-truth G find the set of predictions that have maximum overlap with it (including ties); for each
			prediction in that set, if it is unmatched, then match it to the ground-truth G.

			This function implements the RPN assignment case (i) in Sec. 3.1.2 of :paper:`Faster R-CNN`.
		*/
		void set_low_quality_matches_(const torch::Tensor &match_quality_matrix, torch::Tensor &match_labels);

	private:
		std::vector<float> m_thresholds;
		std::vector<int> m_labels;
		bool m_allow_low_quality_matches;
	};
}