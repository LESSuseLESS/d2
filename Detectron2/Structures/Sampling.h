#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/sampling.py

    /**
		Return `num_samples` (or fewer, if not enough found)
		random samples from `labels` which is a mixture of positives & negatives.
		It will try to return as many positives as possible without
		exceeding `positive_fraction * num_samples`, and then try to
		fill the remaining slots with negatives.

		Args:
			labels (Tensor): (N, ) label vector with values:
				* -1: ignore
				* bg_label: background ("negative") class
				* otherwise: one or more foreground ("positive") classes
			num_samples (int): The total number of labels with value >= 0 to return.
				Values that are not sampled will be filled with -1 (ignore).
			positive_fraction (float): The number of subsampled labels with values > 0
				is `min(num_positives, int(positive_fraction * num_samples))`. The number
				of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
				In order words, if there are not enough positives, the sample is filled with
				negatives. If there are also not enough negatives, then as many elements are
				sampled as is possible.
			bg_label (int): label index of background ("negative") class.

		Returns:
			pos_idx, neg_idx (Tensor):
				1D vector of indices. The total length of both is `num_samples` or fewer.
	*/
	std::tuple<torch::Tensor, torch::Tensor>
		subsample_labels(const torch::Tensor &labels, int num_samples, float positive_fraction, int bg_label);
}