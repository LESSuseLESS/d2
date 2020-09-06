#include "Base.h"
#include "Visualizer.h"

#include "Utils.h"
#include <Detectron2/Structures/RotatedBoxes.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static const int _SMALL_OBJECT_AREA_THRESH = 1000;
static const int _LARGE_MASK_AREA_THRESH = 120000;
static const VisColor _OFF_WHITE = { 1.0, 1.0, 240.0 / 255 };
static const VisColor _BLACK = { 0, 0, 0 };
static const VisColor _RED = { 1.0, 0, 0 };

static const float _KEYPOINT_THRESHOLD = 0.05;

std::vector<std::string> Visualizer::_create_text_labels(const Tensor &classes, const Tensor &scores,
	const std::vector<ClassColor> &class_colors) {
	std::vector<std::string> labels;
	if (classes.numel() && !class_colors.empty()) {
		labels.reserve(classes.size(0));
		for (int i = 0; i < classes.size(0); i++) {
			auto c = classes[i].item<int64_t>();
			assert(c >= 0 && c < class_colors.size());
			labels.push_back(class_colors[c].cls);
		}
	}
	if (scores.numel()) {
		if (labels.empty()) {
			labels.reserve(scores.size(0));
			for (int i = 0; i < scores.size(0); i++) {
				auto s = scores[i].item<float>();
				labels.push_back(FormatString("%.0f%%", s * 100));
			}
		}
		else {
			assert(labels.size() == scores.size(0));
			for (int i = 0; i < labels.size(); i++) {
				auto &l = labels[i];
				auto s = scores[i].item<float>();
				l += FormatString(" %.0f%%", s * 100);
			}
		}
	}
	return labels;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Visualizer::Visualizer(const torch::Tensor &img_rgb, Metadata metadata, float scale, ColorMode instance_mode) :
	m_img(img_rgb.clamp(0, 255).to(torch::kUInt8)),
	m_metadata(metadata),
	m_output(m_img, scale),
	m_cpu_device(torch::kCPU),
	m_default_font_size(max(	// too small texts are useless, therefore clamp to 9
		(float)sqrt(m_output.height() * m_output.width()) / 90, 10 / scale)),
	m_instance_mode(instance_mode)
{
}

void Visualizer::set_grayscale_image(const torch::Tensor &img) {
	m_output.img() = _create_grayscale_image(img);
}

VisImage Visualizer::draw_instance_predictions(const InstancesPtr &predictions) {
	Tensor classes;
	if (predictions->has("pred_classes")) {
		classes = predictions->getTensor("pred_classes");
	}
	vector<VisColor> colors;
	float alpha = 0.5;
	if (m_instance_mode == ColorMode::kSEGMENTATION && !m_metadata->thing.empty()) {
		colors.reserve(classes.size(0));
		for (int i = 0; i < classes.size(0); i++) {
			auto c = classes[i].item<int64_t>();
			VisColor xs = color_normalize(m_metadata->thing[c].color);
			colors.push_back(color_jitter(xs));
		}
		alpha = 0.8;
	}

	if (m_instance_mode == ColorMode::kIMAGE_BW) {
		set_grayscale_image(predictions->getTensor("pred_masks").any(0) > 0);
		alpha = 0.3;
	}

	Tensor boxes; if (predictions->has("pred_boxes")) boxes = predictions->getTensor("pred_boxes");
	Tensor scores; if (predictions->has("scores")) scores = predictions->getTensor("scores");
	auto labels = _create_text_labels(classes, scores, m_metadata->thing);
	Tensor keypoints; if (predictions->has("pred_keypoints")) keypoints = predictions->getTensor("pred_keypoints");
	vector<shared_ptr<GenericMask>> masks;
	if (predictions->has("pred_masks")) {
		auto t_masks = predictions->getTensor("pred_masks");
		int count = t_masks.size(0);
		masks.reserve(count);
		for (int i = 0; i < count; i++) {
			auto mask = make_shared<GenericMask>(t_masks[i], m_output.height(), m_output.width());
			masks.push_back(mask);
		}
	}

	overlay_instances(boxes, labels, masks, keypoints, colors, alpha);
	return m_output;
}

VisImage Visualizer::draw_sem_seg(const torch::Tensor &sem_seg, int area_threshold, float alpha) {
	Tensor labels, _dummy_, areas;
	tie(labels, _dummy_, areas) = torch::_unique2(sem_seg, true, false, true);
	auto sorted_idxs = tolist(torch::argsort(-areas));
	labels = labels.index({ sorted_idxs });
	for (int i = 0; i < labels.size(0); i++) {
		auto label = labels[i].item<int64_t>();
		if (label >= 0 && label < m_metadata->stuff.size()) {
			VisColor mask_color = color_normalize(m_metadata->stuff[label].color);

			auto binary_mask = (sem_seg == label).to(torch::kUInt8);
			auto &text = m_metadata->stuff[label].cls;
			draw_binary_mask(binary_mask, mask_color, _OFF_WHITE, text, alpha, area_threshold);
		}
	}
	return m_output;
}

VisImage Visualizer::draw_panoptic_seg_predictions(const torch::Tensor &panoptic_seg,
	const std::vector<SegmentInfo> &segments_info, int area_threshold, float alpha) {
	_PanopticPrediction pred(panoptic_seg, segments_info);

	if (m_instance_mode == ColorMode::kIMAGE_BW) {
		set_grayscale_image(pred.non_empty_mask());
	}

	// draw mask for all semantic segments first i.e. "stuff"
	pred.semantic_masks([=](torch::Tensor mask, const SegmentInfo &sinfo) {
		auto category_idx = sinfo.category_id;
		assert(category_idx >= 0 && category_idx < m_metadata->stuff.size());
		VisColor mask_color = color_normalize(m_metadata->stuff[category_idx].color);
		auto &text = m_metadata->stuff[category_idx].cls;
		draw_binary_mask(mask, mask_color, _OFF_WHITE, text, alpha, area_threshold);
		});

	// draw mask for all instances second
	vector<shared_ptr<GenericMask>> masks;
	vector<int64_t> category_ids;
	vector<float> scores;
	vector<VisColor> colors;
	bool has_instance = false;
	pred.instance_masks([&](torch::Tensor mask, const SegmentInfo &sinfo) {
		has_instance = true;
		masks.push_back(make_shared<GenericMask>(mask, m_output.height(), m_output.width()));
		category_ids.push_back(sinfo.category_id);
		scores.push_back(sinfo.score);
		colors.push_back(color_random());
		});
	if (has_instance) {
		auto labels = _create_text_labels(torch::tensor(category_ids), torch::tensor(scores), m_metadata->thing);
		overlay_instances(Tensor(), labels, masks, {}, colors, alpha);
	}
	return m_output;
}

/*~!
def draw_dataset_dict(self, dic):
    annos = dic.get("annotations", None)
    if annos:
        if "segmentation" in annos[0]:
            masks = [x["segmentation"] for x in annos]
        else:
            masks = None
        if "keypoints" in annos[0]:
            keypts = [x["keypoints"] for x in annos]
            keypts = np.array(keypts).reshape(len(annos), -1, 3)
        else:
            keypts = None

        boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) for x in annos]

        labels = [x["category_id"] for x in annos]
        colors = None
        if m_instance_mode == ColorMode.SEGMENTATION and m_metadata->get("thing_colors"):
            colors = [
                m_jitter([x / 255 for x in m_metadata->thing_colors[c]]) for c in labels
            ]
        names = m_metadata->get("thing_classes", None)
        if names:
            labels = [names[i] for i in labels]
        labels = [
            "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
            for i, a in zip(labels, annos)
        ]
        m_overlay_instances(
            labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
        )

    sem_seg = dic.get("sem_seg", None)
    if sem_seg is None and "sem_seg_file_name" in dic:
        with PathManager.open(dic["sem_seg_file_name"], "rb") as f:
            sem_seg = Image.open(f)
            sem_seg = np.asarray(sem_seg, dtype="uint8")
    if sem_seg is not None:
        m_draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
    return m_output
*/

VisImage Visualizer::overlay_instances(const torch::Tensor &boxes, const std::vector<std::string> &labels,
	const std::vector<std::shared_ptr<GenericMask>> &masks, const torch::Tensor &keypoints,
	const std::vector<VisColor> &assigned_colors, float alpha) {
	int num_instances = 0;
	if (boxes.defined() && boxes.size(0)) {
		num_instances = boxes.size(0);
	}
	if (masks.size()) {
		assert(num_instances == 0 || masks.size() == num_instances);
		num_instances = masks.size();
	}
	if (keypoints.defined() && keypoints.size(0)) {
		assert(num_instances == 0 || keypoints.size(0) == num_instances);
		num_instances = keypoints.size(0);
	}
	assert(labels.empty() || labels.size() == num_instances);
	if (num_instances == 0) {
		return m_output;
	}

	std::vector<VisColor> colors = assigned_colors;
	if (colors.empty()) {
		colors.reserve(num_instances);
		for (int i = 0; i < num_instances; i++) {
			colors.push_back(color_random());
		}
	}
	if (boxes.defined() && boxes.size(0) && boxes.size(1) == 5) {
		return overlay_rotated_instances(boxes, labels, colors);
	}

	// Display in largest to smallest order to reduce occlusion.
	Tensor areas;
	if (boxes.defined() && boxes.size(0)) {
		areas = torch::prod(boxes.index({ Colon, Slice(2, None) }) - boxes.index({ Colon, Slice(None, 2) }), 1);
	}
	else if (masks.size()) {
		vector<float> areas_vec;
		areas_vec.reserve(masks.size());
		for (auto x : masks) {
			areas_vec.push_back(x->area());
		}
		areas = torch::tensor(areas_vec);
	}

	Tensor sorted_boxes;
	vector<string> sorted_labels;
	vector<shared_ptr<GenericMask>> sorted_masks;
	vector<VisColor> sorted_colors;
	Tensor sorted_keypoints;
	if (areas.size(0)) {
		auto sorted_idxs = tolist(torch::argsort(-areas));
		// Re-order overlapped instances in descending order.
		if (boxes.defined() && boxes.size(0)) {
			sorted_boxes = boxes.index({ sorted_idxs });
		}
		if (!labels.empty()) {
			vec_select(sorted_labels, labels, sorted_idxs);
		}
		if (!masks.empty()) {
			vec_select(sorted_masks, masks, sorted_idxs);
		}
		if (!assigned_colors.empty()) {
			vec_select(sorted_colors, assigned_colors, sorted_idxs);
		}
		if (keypoints.defined() && keypoints.size(0)) {
			sorted_keypoints = keypoints.index({ sorted_idxs });
		}
	}
	else {
		sorted_boxes = boxes;
		sorted_labels = labels;
		sorted_masks = masks;
		sorted_colors = assigned_colors;
		sorted_keypoints = keypoints;
	}

	for (int i = 0; i < num_instances; i++) {
		auto &color = sorted_colors[i];
		if (sorted_boxes.defined() && sorted_boxes.size(0)) {
			draw_box(sorted_boxes[i], 0.5, color);
		}
		if (sorted_masks.size()) {
			for (auto &segment : sorted_masks[i]->polygons()) {
				draw_polygon(segment.reshape({ -1, 2 }), color, {}, alpha);
			}
		}
		if (sorted_labels.size()) {
			int x0, y0, x1, y1;
			Pos text_pos;
			Canvas::Alignment horiz_align;
			// first get a box
			if (sorted_boxes.defined() && sorted_boxes.size(0)) {
				tie(x0, y0, x1, y1) = Boxes::boxes(sorted_boxes)->bbox(i);
				text_pos = { x0, y0 };  // if drawing boxes, put text on the box corner.
				horiz_align = Canvas::kLeft;
			}
			else if (sorted_masks.size()) {
				auto b = sorted_masks[i]->bbox();
				x0 = b[0].item<int>();
				y0 = b[1].item<int>(); 
				x1 = b[2].item<int>(); 
				y1 = b[3].item<int>();

				// draw text in the center (defined by median) when box is not drawn
				// median is less sensitive to outliers.
				auto median = get<0>(torch::median(sorted_masks[i]->mask().nonzero(), 0));
				text_pos = { (int)median[1].item<int64_t>(), (int)median[0].item<int64_t>() };
				horiz_align = Canvas::kCenter;
			}
			else {
				continue; // drawing the box confidence for keypoints isn't very useful.
			}
			// for small objects, draw text at the side to avoid occlusion
			auto instance_area = (y1 - y0) * (x1 - x0);
			if (instance_area < _SMALL_OBJECT_AREA_THRESH * m_output.scale() || (y1 - y0) < 40 * m_output.scale()) {
				if (y1 >= m_output.height() - 5) {
					text_pos = { x1, y0 };
				}
				else {
					text_pos = { x0, y1 };
				}
			}

			auto height_ratio = (y1 - y0) / sqrt(m_output.height() * m_output.width());
			auto lighter_color = color_brightness(color, 0.7);
			auto font_size = clip<float>((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * m_default_font_size;
			draw_text(sorted_labels[i], text_pos, font_size, lighter_color, horiz_align);
		}
	}

	// draw keypoints
	if (sorted_keypoints.defined() && sorted_keypoints.size(0)) {
		for (int i = 0; i < sorted_keypoints.size(0); i++) {
			auto keypoints_per_instance = sorted_keypoints[i];
			draw_and_connect_keypoints(keypoints_per_instance);
		}
	}
	return m_output;
}

VisImage Visualizer::overlay_rotated_instances(const torch::Tensor &boxes,
	const std::vector<std::string> &labels, const std::vector<VisColor> &assigned_colors) {
	int num_instances = boxes.size(0);
	if (num_instances == 0) {
		return m_output;
	}

	std::vector<VisColor> colors = assigned_colors;
	if (colors.empty()) {
		colors.reserve(num_instances);
		for (int i = 0; i < num_instances; i++) {
			colors.push_back(color_random());
		}
	}

	Tensor sorted_boxes;
	vector<string> sorted_labels;
	vector<VisColor> sorted_colors;

	// Display in largest to smallest order to reduce occlusion.
	Tensor areas = boxes.index({ Colon, 2 }) * boxes.index({ Colon,3 });
	auto sorted_idxs = tolist(torch::argsort(-areas));
	// Re-order overlapped instances in descending order.
	sorted_boxes = boxes.index({ sorted_idxs });
	vec_select(sorted_labels, labels, sorted_idxs);
	vec_select(sorted_colors, colors, sorted_idxs);

	for (int i = 0; i < num_instances; i++) {
		draw_rotated_box_with_label(sorted_boxes[i], 0.5, sorted_colors[i], Canvas::kSolid,
			sorted_labels.size() ? sorted_labels[i] : "");
	}
	return m_output;
}

VisImage Visualizer::draw_and_connect_keypoints(const torch::Tensor &keypoints) {
	map<string, Pos> visible;
	auto &keypoint_names = m_metadata->keypoint_names;
	int count = keypoints.size(0);
	for (int idx = 0; idx < count; idx++) {
		auto keypoint = keypoints[idx];
		// draw keypoint
		int x = (int)keypoint[0].item<float>();
		int y = (int)keypoint[1].item<float>();
		auto prob = keypoint[2].item<float>();
		if (prob > _KEYPOINT_THRESHOLD) {
			draw_circle({ x, y }, _RED);
			if (!keypoint_names.empty()) {
				assert(idx >= 0 && idx < keypoint_names.size());
				auto &keypoint_name = keypoint_names[idx];
				visible[keypoint_name] = { x, y };
			}
		}
	}
	if (!m_metadata->keypoint_connection_rules.empty()) {
		for (auto rule : m_metadata->keypoint_connection_rules) {
			string kp0, kp1; VisColor color;
			tie(kp0, kp1, color) = rule;
			if (visible.find(kp0) != visible.end() && visible.find(kp1) != visible.end()) {
				auto p0 = visible[kp0];
				auto p1 = visible[kp1];
				color = color_normalize(color);
				draw_line({ p0.x, p1.x }, { p0.y, p1.y }, color);
			}
		}
	}

	// draw lines from nose to mid-shoulder and mid-shoulder to mid-hip
	// Note that this strategy is specific to person keypoints.
	// For other keypoints, it should just do nothing
	if (visible.find("left_shoulder") != visible.end() && visible.find("right_shoulder") != visible.end()) {
		auto ls = visible["left_shoulder"];
		auto rs = visible["right_shoulder"];
		auto mid_shoulder_x = (ls.x + rs.x) / 2;
		auto mid_shoulder_y = (ls.y + rs.y) / 2;

		// draw line from nose to mid-shoulder
		Pos nose;
		if (visible.find("nose") != visible.end()) {
			nose = visible["nose"];
			draw_line({ nose.x, mid_shoulder_x }, { nose.y, mid_shoulder_y }, _RED);
		}

		if (visible.find("left_hip") != visible.end() && visible.find("right_hip") != visible.end()) {
			int lh_x, lh_y, rh_x, rh_y;
			// draw line from mid-shoulder to mid-hip
			auto lh = visible["left_hip"];
			auto rh = visible["right_hip"];
			auto mid_hip_x = (lh.x + rh.x) / 2;
			auto mid_hip_y = (lh.y + rh.y) / 2;
			draw_line({ mid_hip_x, mid_shoulder_x }, { mid_hip_y, mid_shoulder_y }, _RED);
		}
	}
	return m_output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Primitive drawing functions

VisImage Visualizer::draw_text(const std::string &text, const Pos &position, float font_size,
	const VisColor &color, const Canvas::Alignment &horizontal_alignment, float rotation) {
	if (font_size <= 0) {
		font_size = m_default_font_size;
	}

	// since the text background is dark, we don't want the text to be dark
	Tensor t_color = torch::tensor(color_at_least(color, 0.2));
	auto maximum = max(0.8f, max(t_color).item<float>());
	t_color.index_put_(argmax(t_color), torch::tensor(maximum));
	VisColor updated(color);
	for (int i = 0; i < color.size(); i++) {
		updated[i] = t_color[i].item<float>();
	}

	Canvas::DrawTextOptions options;
	options.font_size = font_size * m_output.scale();
	options.font_family = "sans-serif";
	options.font_color = updated;
	options.bbox_color = _BLACK;
	options.bbox_alpha = 0.8;
	options.bbox_padding = 0.7;
	options.edge_color = {};
	options.vertical_alignment = Canvas::kTop;
	options.horizontal_alignment = horizontal_alignment;
	options.zorder = 10;
	options.rotation = rotation;
	m_output.get_canvas()->DrawText(position.x, position.y, text, options);
	return m_output;
}

VisImage Visualizer::draw_box(const torch::Tensor &box_coord, float alpha, const VisColor &edge_color,
	const Canvas::LineStyle &line_style) {
	int x0, y0, x1, y1;
	tie(x0, y0, x1, y1) = Boxes::boxes(box_coord.reshape({ -1,4 }))->bbox(0);
	auto width = x1 - x0;
	auto height = y1 - y0;

	auto linewidth = max(m_default_font_size / 4, 1.0f);

	Canvas::DrawRectangleOptions options;
	options.fill = false;
	options.edge_color = edge_color;
	options.line_width = linewidth * m_output.scale();
	options.alpha = alpha;
	options.line_style = line_style;
	m_output.get_canvas()->DrawRectangle(x0, y0, width, height, options);
	return m_output;
}

VisImage Visualizer::draw_rotated_box_with_label(const torch::Tensor &rotated_box, float alpha,
	const VisColor &edge_color, const Canvas::LineStyle &line_style, const std::string &label) {
	int cnt_x, cnt_y, w, h;
	tie(cnt_x, cnt_y, w, h) = Boxes::boxes(rotated_box)->bbox(0);
	float angle = rotated_box[4].item<float>();
	auto area = w * h;
	// use thinner lines when the box is small
	auto linewidth = m_default_font_size / (area < _SMALL_OBJECT_AREA_THRESH * m_output.scale() ? 6 : 3);

	auto theta = angle * M_PI / 180.0;
	auto c = cos(theta);
	auto s = sin(theta);
	vector<vector<int>> rect = { { -w / 2, h / 2 }, { -w / 2, -h / 2 }, { w / 2, -h / 2 }, { w / 2, h / 2 } };
	// x: left->right ; y: top->down
	vector<vector<int>> rotated_rect;
	for (auto pt : rect) {
		auto xx = (double)pt[0];
		auto yy = (double)pt[1];
		rotated_rect.push_back({ (int)(s * yy + c * xx + cnt_x), (int)(c * yy - s * xx + cnt_y) });
	}
	for (int k = 0; k < 4; k++) {
		auto j = (k + 1) % 4;
		draw_line({ rotated_rect[k][0], rotated_rect[j][0] }, { rotated_rect[k][1], rotated_rect[j][1] },
			edge_color, (k == 1 ? Canvas::kDashed : Canvas::kSolid), linewidth);
	}
	if (!label.empty()) {
		auto text_pos = rotated_rect[1];  // topleft corner

		auto height_ratio = h / sqrt(m_output.height() * m_output.width());
		auto label_color = color_brightness(edge_color, 0.7);
		auto font_size = (clip<float>((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * m_default_font_size);
		draw_text(label, { text_pos[0], text_pos[1] }, font_size, label_color, Canvas::kCenter, angle);
	}
	return m_output;
}

VisImage Visualizer::draw_circle(const Pos &circle_coord, const VisColor &color, int radius) {
	Canvas::DrawCircleOptions options;
	options.fill = true;
	options.color = color;
	m_output.get_canvas()->DrawCircle(circle_coord.x, circle_coord.y, radius, options);
	return m_output;
}

VisImage Visualizer::draw_line(const std::vector<int> &x_data, const std::vector<int> &y_data,
	const VisColor &color, const Canvas::LineStyle &linestyle, float linewidth) {
	if (linewidth <= 0) {
		linewidth = m_default_font_size / 3;
	}
	linewidth = max(linewidth, 1.0f);

	Canvas::DrawLine2DOptions options;
	options.line_width = linewidth * m_output.scale();
	options.color = color;
	options.line_style = linestyle;
	m_output.get_canvas()->DrawLine2D(x_data, y_data, options);
	return m_output;
}

VisImage Visualizer::draw_binary_mask(torch::Tensor binary_mask, VisColor color,
	const VisColor &edge_color, const std::string &text, float alpha, float area_threshold) {
	if (color.empty()) {
		color = color_random();
	}
	if (area_threshold <= 0) {
		area_threshold = 4096;
	}

	bool has_valid_segment = false;
	binary_mask = binary_mask.to(torch::kUInt8);  // opencv needs uint8
	GenericMask mask(binary_mask, m_output.height(), m_output.width());
	Size2D shape2d = { binary_mask.size(0), binary_mask.size(1) };

	if (!mask.has_holes()) {
		// draw polygons for regular masks
		for (auto segment : mask.polygons()) {
			auto mo = mask_util::frPyObjects_single(segment, shape2d.height, shape2d.width);
			auto area = mask_util::area_single(mo);
			if (area.item<float>() < area_threshold) {
				continue;
			}
			bool has_valid_segment = true;
			segment = segment.reshape({ -1, 2 });
			draw_polygon(segment, color, edge_color, alpha);
		}
	}
	else {
		auto rgba = torch::zeros({ shape2d.height, shape2d.width, 4 }, torch::kFloat32);
		rgba.index_fill_(2, torch::tensor({ 0 }), color[0]);
		rgba.index_fill_(2, torch::tensor({ 1 }), color[1]);
		rgba.index_fill_(2, torch::tensor({ 2 }), color[2]);
		auto m = (mask.mask() == 1).to(torch::kFloat32) * alpha;
		rgba.index_put_({ Colon, Colon, Slice(3) }, m.reshape({ shape2d.height, shape2d.width, 1 }));
		has_valid_segment = true;
		m_output.get_canvas()->DrawImage(rgba);
	}

	if (!text.empty() && has_valid_segment) {
		// TODO sometimes drawn on wrong objects. the heuristics here can improve.
		auto lighter_color = color_brightness(color, 0.7);
		cv::Mat cc_labels, stats, centroids;
		auto _num_cc = cv::connectedComponentsWithStats(image_to_mat(binary_mask), cc_labels, stats, centroids, 8);
		auto t_stats = mat_to_tensor(stats);
		auto largest_component_id = (int)argmax(t_stats.index({ Slice(1, None), -1 })).item<int64_t>() + 1;

		// draw text on the largest component, as well as other very large components.
		for (int cid = 1; cid < _num_cc; cid++) {
			if (cid == largest_component_id ||
				(t_stats.index({ cid, -1 }) > _LARGE_MASK_AREA_THRESH).all().item<bool>()) {
				// median is more stable than centroid
				// center = centroids[largest_component_id]
				auto median = get<0>(torch::median((mat_to_tensor(cc_labels) == cid).nonzero(), 0));
				Pos text_pos = { (int)median[1].item<int64_t>(), (int)median[0].item<int64_t>() };
				draw_text(text, text_pos, 0.0f, lighter_color);
			}
		}
	}
	return m_output;
}

VisImage Visualizer::draw_polygon(const torch::Tensor &segment, VisColor color, VisColor edge_color, float alpha) {
	if (edge_color.empty()) {
		// make edge color darker than the polygon color
		if (alpha > 0.8) {
			edge_color = color_brightness(color, -0.7);
		}
		else {
			edge_color = color;
		}
	}
	edge_color.push_back(1);
	color.push_back(alpha);

	Canvas::DrawPolygonOptions options;
	options.fill = true;
	options.face_color = color;
	options.edge_color = edge_color;
	options.line_width = max(m_default_font_size / 15 * m_output.scale(), 1.0f);
	m_output.get_canvas()->DrawPolygon(segment, options);
	return m_output;
}

torch::Tensor Visualizer::_create_grayscale_image(const torch::Tensor &mask) {
	auto img_bw = m_img.to(torch::kFloat32).mean(2);
	img_bw = torch::stack({ img_bw, img_bw, img_bw }, 2);
	if (mask.numel()) {
		img_bw[mask] = m_img[mask];
	}
	return img_bw;
}
