#include "Base.h"
#include "ResizeShortestEdge.h"

#include "ResizeTransform.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ResizeShortestEdge::ResizeShortestEdge(int short_edge_length, int64_t max_size, const std::string &sample_style,
	Transform::Interp interp) :
	ResizeShortestEdge({ short_edge_length, short_edge_length }, max_size, sample_style, interp) {
}

ResizeShortestEdge::ResizeShortestEdge(const std::vector<int> &short_edge_length, int64_t max_size,
	const std::string &sample_style, Transform::Interp interp) :
	m_short_edge_length(short_edge_length),
	m_max_size(max_size),
	m_is_range(sample_style == "range"),
	m_interp(interp)
{
	assert(sample_style == "range" || sample_style == "choice");
}

std::shared_ptr<Transform> ResizeShortestEdge::get_transform(torch::Tensor img) {
	auto h = img.size(0);
	auto w = img.size(1);

	int64_t size;
	if (m_is_range) {
		size = torch::randint(m_short_edge_length[0], m_short_edge_length[1] + 1, 1).item<int64_t>();
	}
	else {
		size = m_short_edge_length[torch::randint(0, m_short_edge_length.size(), 1).item<int64_t>()];
	}
	if (size == 0) {
		return make_shared<NoOpTransform>();
	}

	auto scale = (float)size / min(h, w);
	float newh, neww;
	if (h < w) {
		newh = size;
		neww = scale * w;
	}
	else {
		newh = scale * h;
		neww = size;
	}
	if (max(newh, neww) > m_max_size) {
		scale = (float)m_max_size / max(newh, neww);
		newh = newh * scale;
		neww = neww * scale;
	}
	return make_shared<ResizeTransform>(h, w, int(newh + 0.5), int(neww + 0.5), m_interp);
}
