#include "Base.h"
#include "VisImage.h"
#include "Utils.h"
#include "cvCanvas.h"

using namespace std;
using namespace cv;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

_PanopticPrediction::_PanopticPrediction(const torch::Tensor &panoptic_seg,
	const std::vector<SegmentInfo> &segments_info) : m_seg(panoptic_seg) {
	for (auto s : segments_info) {
		m_sinfo[s.id] = s;
	}

	Tensor segment_ids, _dummy_, areas;
	tie(segment_ids, _dummy_, areas) = torch::_unique2(panoptic_seg, true, false, true);
	auto sorted_idxs = torch::argsort(-areas);
	m_seg_ids = segment_ids.index({ sorted_idxs });
	m_seg_areas = areas.index({ sorted_idxs });
	m_seg_ids = tolist(m_seg_ids);
	int count = m_seg_ids.size(0);
	assert(m_seg_areas.size(0) == count);
	for (int i = 0; i < count; i++) {
		auto sid = m_seg_ids[i].item<int64_t>();
		auto iter = m_sinfo.find(sid);
		if (iter != m_sinfo.end()) {
			iter->second.area = m_seg_areas[i].item<float>();
		}
	}
}

torch::Tensor _PanopticPrediction::non_empty_mask() const {
	vector<int64_t> empty_ids;
	int count = m_seg_ids.size(0);
	for (int i = 0; i < count; i++) {
		auto id = m_seg_ids[i].item<int64_t>();
		if (m_sinfo.find(id) == m_sinfo.end()) {
			empty_ids.push_back(id);
		}
	}
	if (empty_ids.empty()) {
		return torch::zeros(m_seg.sizes(), torch::kUInt8);
	}
	assert(empty_ids.size() == 1); // >1 ids corresponds to no labels. This is currently not supported
	return (m_seg != empty_ids[0]);
}

void _PanopticPrediction::semantic_masks(std::function<void(torch::Tensor, const SegmentInfo &)> func) const {
	int count = m_seg_ids.size(0);
	for (int i = 0; i < count; i++) {
		auto sid = m_seg_ids[i].item<int64_t>();
		auto iter = m_sinfo.find(sid);
		if (iter == m_sinfo.end() || iter->second.isthing) {
			// Some pixels (e.g. id 0 in PanopticFPN) have no instance or semantic predictions.
			continue;
		}
		func((m_seg == sid), iter->second);
	}
}

void _PanopticPrediction::instance_masks(std::function<void(torch::Tensor, const SegmentInfo &)> func) const {
	int count = m_seg_ids.size(0);
	for (int i = 0; i < count; i++) {
		auto sid = m_seg_ids[i].item<int64_t>();
		auto iter = m_sinfo.find(sid);
		if (iter == m_sinfo.end() || iter->second.isthing) {
			// Some pixels (e.g. id 0 in PanopticFPN) have no instance or semantic predictions.
			continue;
		}
		auto mask = (m_seg == sid);
		if (mask.sum().item<int64_t>() > 0) {
			func(mask, iter->second);
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VisImage::VisImage(const torch::Tensor &img, float scale) :
	m_img(img), m_scale(scale), m_height(img.size(0)), m_width(img.size(1)) {
	_setup_figure();
}

void VisImage::_setup_figure() {
	m_canvas = make_shared<cvCanvas>(m_height * m_scale, m_width * m_scale);
	/*~!
	fig = mplfigure.Figure(frameon=False)
	m_dpi = fig.get_dpi()
	// add a small 1e-2 to avoid precision lost due to matplotlib's truncation
	// (https://github.com/matplotlib/matplotlib/issues/15363)
	fig.set_size_inches(
		(m_width * m_scale + 1e-2) / m_dpi,
		(m_height * m_scale + 1e-2) / m_dpi,
	)
	m_canvas = FigureCanvasAgg(fig)
	// m_canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
	ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
	ax.axis("off")
	ax.set_xlim(0.0, m_width)
	ax.set_ylim(m_height)

	m_fig = fig
	m_ax = ax
	*/
}

void VisImage::save(const std::string &filepath) const {
	string lowered = lower(filepath);
	if (endswith(lowered, ".jpg") || endswith(lowered, ".png")) {
		// faster than matplotlib's imshow
		auto image = torch::flip(get_image(), { -1 });
		imwrite(filepath, image_to_mat(image));
	}
	else {
		assert(false);
		/*~!
		// support general formats (e.g. pdf)
		m_ax.imshow(m_img, interpolation = "nearest");
		m_fig.savefig(filepath);
		*/
	}
}

torch::Tensor VisImage::get_image() const {
	Tensor buffer; int height, width;
	tie(buffer, height, width) = m_canvas->SaveToTensor();

	auto img_rgba = buffer.reshape({ height, width, 4 });
	auto splitted = torch::split(img_rgba, { 3 }, 2);
	auto rgb = splitted[0];
	auto alpha = splitted[1].to(torch::kFloat32) / 255;

	auto img = m_img;
	if (m_width != width || m_height != height) {
		auto mat_img = image_to_mat(m_img);
		cv::resize(mat_img, mat_img, { width, height });
		img = image_to_tensor(mat_img);
	}

	// imshow is slow. blend manually (still quite slow)
	auto visualized_image = img * (1 - alpha) + rgb * alpha;
	visualized_image = visualized_image.to(torch::kUInt8);
	return visualized_image;
}
