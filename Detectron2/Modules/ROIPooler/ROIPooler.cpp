#include "Base.h"
#include "ROIPooler.h"

#include <Detectron2/Structures/Boxes.h>
#include "ROIAlign.h"
#include "ROIAlignRotated.h"
#include "ROIPool.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Tensor ROIPoolerImpl::assign_boxes_to_levels(const BoxesList &box_lists, int min_level, int max_level,
	int canonical_box_size, int canonical_level) {
	TensorVec areas;
	areas.reserve(box_lists.size());
	for (auto &boxes : box_lists) {
		areas.push_back(Boxes::boxes(boxes)->area());
	}
	auto box_sizes = torch::sqrt(torch::cat(areas));

	// Eqn.(1) in FPN paper
	auto eps = numeric_limits<float>::epsilon();
	auto level_assignments = torch::floor(canonical_level + torch::log2(box_sizes / canonical_box_size + eps));

	// clamp level to (min, max), in case the box size is too large or too small for the available feature maps
	level_assignments.clamp_(min_level, max_level);
	return level_assignments.to(torch::kInt64) - min_level;
}

Tensor ROIPoolerImpl::convert_boxes_to_pooler_format(const BoxesList &box_lists) {
	auto fmt_box_list = [](const Tensor &box_tensor, int batch_index) {
		auto repeated_index = torch::full({ box_tensor.size(0), 1 }, batch_index,
			dtype(box_tensor.dtype()).device(box_tensor.device()));
		return torch::cat({ repeated_index, box_tensor }, 1);
	};
	TensorVec pooler_fmt_boxes;
	int index = 0;
	for (auto &boxes : box_lists) {
		pooler_fmt_boxes.push_back(fmt_box_list(boxes, index++));
	}
	return torch::cat(pooler_fmt_boxes);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ROIPoolerImpl::ROIPoolerImpl(const std::string &pooler_type, const Size2D &output_size,
	const vector<float> &scales, int sampling_ratio, int canonical_box_size, int canonical_level) :
	m_output_size(output_size),
	m_canonical_box_size(canonical_box_size),
	m_canonical_level(canonical_level)
{
	assert(m_canonical_box_size > 0);

	for (int i = 0; i < scales.size(); i++) {
		auto scale = scales[i];
		shared_ptr<ROIPoolerLevelImpl> level_pooler;
		if (pooler_type == "ROIAlign") {
			level_pooler = make_shared<ROIAlignImpl>(m_output_size, scale, sampling_ratio, false);
		}
		else if (pooler_type == "ROIAlignV2") {
			level_pooler = make_shared<ROIAlignImpl>(m_output_size, scale, sampling_ratio, true);
		}
		else if (pooler_type == "ROIPool") {
			level_pooler = make_shared<RoIPoolImpl>(m_output_size, scale); break;
		}
		else if (pooler_type == "ROIAlignRotated") {
			level_pooler = make_shared<ROIAlignRotatedImpl>(m_output_size, scale, sampling_ratio);
		} else {
			assert(false);
			break;
		}
		register_module(FormatString("level_poolers%d", i), level_pooler); // ouch: this is a guess on ModuleList
		m_level_poolers.push_back(level_pooler);
	}

	// Map scale (defined as 1 / stride) to its feature map level under the assumption that stride is a power of 2.
	auto min_level_ = -(log2(scales[0]));
	auto max_level_ = -(log2(scales[scales .size() - 1]));
	assert(abs(min_level_ - float(int(min_level_))) <= 0.000001f);
	assert(abs(max_level_ - float(int(max_level_))) <= 0.000001f);
	m_min_level = int(min_level_);
	m_max_level = int(max_level_);
	assert(scales.size() == m_max_level - m_min_level + 1); // sizes of input featuremaps need to form a pyramid
	assert(0 < m_min_level && m_min_level <= m_max_level);
}

Tensor ROIPoolerImpl::forward(const TensorVec &x, const BoxesList &box_lists) {
	auto num_level_assignments = m_level_poolers.size();
	assert(x.size() == num_level_assignments);
	assert(box_lists.size() == x[0].size(0));

	auto pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists);
	if (num_level_assignments == 1) {
		return m_level_poolers[0]->forward(x[0], pooler_fmt_boxes);
	}

	auto level_assignments = assign_boxes_to_levels(box_lists, m_min_level, m_max_level,
		m_canonical_box_size, m_canonical_level);

	auto num_boxes = pooler_fmt_boxes.size(0);
	auto num_channels = x[0].size(1);
	auto output_height = m_output_size.height;

	auto output = torch::zeros(
		{ num_boxes, num_channels, output_height, output_height },
		dtype(x[0].dtype()).device(x[0].device()));

	for (int level = 0; level < num_level_assignments; level++) {
		auto x_level = x[level];
		auto pooler = m_level_poolers[level];
		auto inds = torch::nonzero(level_assignments == level).index({ Colon, 0 });
		auto pooler_fmt_boxes_level = pooler_fmt_boxes.index(inds);
		output.index_put_({ inds }, pooler->forward(x_level, pooler_fmt_boxes_level));
	}

	return output;
}
