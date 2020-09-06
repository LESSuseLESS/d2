#include "Base.h"
#include "cvCanvas.h"
#include "Utils.h"
#include "VisColor.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
// NOTE: "All the functions include the parameter color that uses an RGB value"
cv::Scalar cvCanvas::cvColor(const std::vector<float> &c) {
	auto d = color_denormalize(c);
	switch (d.size()) {
	case 3:
		return cv::Scalar(d[0], d[1], d[2], 255);
	case 4:
		return cv::Scalar(d[0], d[1], d[2], d[3]);
	default:
		assert(false);
		break;
	}
	return {};
}

cv::Scalar cvCanvas::cvColor(const Color3 &c, float alpha) {
	auto d = color_denormalize(c);
	return cv::Scalar(d[0], d[1], d[2], alpha * 255);
}

int cvCanvas::cvLineWidth(float line_width) {
	return (int)line_width;
}

int cvCanvas::cvLineType(LineStyle line_style) {
	// ouch: this isn't really the same:
	switch (line_style) {
	case kSolid:	return cv::FILLED;
	case kDotted:	return cv::LINE_4;
	case kDashed:	return cv::LINE_8;
	case kDashDot:	return cv::LINE_AA;
	default: assert(false); break;
	}
	return 0;
}

double cvCanvas::cvFontScale(float font_size) {
	return font_size / 20;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

cvCanvas::cvCanvas(int height, int width) : m_canvas(height, width, CV_8UC4, { 0, 0, 0, 0 }) {
}

std::tuple<torch::Tensor, int, int> cvCanvas::SaveToTensor() {
	return { image_to_tensor(m_canvas), m_canvas.size().height, m_canvas.size().width };
}

void cvCanvas::DrawLine2D(const std::vector<int> &x_data, const std::vector<int> &y_data,
	const DrawLine2DOptions &options) {
	int count = x_data.size();
	assert(count > 1);
	assert(y_data.size() == count);
	auto xLast = x_data[0]; auto yLast = y_data[0];
	for (int i = 1; i < x_data.size(); i++) {
		auto x = x_data[i]; auto y = y_data[i];
		cv::line(m_canvas, { xLast, yLast }, { x, y }, cvColor(options.color),
			cvLineWidth(options.line_width), cvLineType(options.line_style));
		xLast = x; yLast = y;
	}
}

void cvCanvas::DrawRectangle(int x, int y, int width, int height, const DrawRectangleOptions &options) {
	cv::rectangle(m_canvas, { x, y, width, height }, cvColor(options.edge_color, options.alpha),
		options.fill ? -1 : cvLineWidth(options.line_width), cvLineType(options.line_style));
}

void cvCanvas::DrawPolygon(const torch::Tensor &segment, const DrawPolygonOptions &options) {
	int npts = segment.size(0);
	assert(npts > 2);
	std::unique_ptr<cv::Point[]> pts{ new cv::Point[npts] };
	for (int i = 0; i < npts; i++) {
		auto point = segment[i];
		pts[i] = { point[0].item<int>(), point[1].item<int>() };
	}
	auto ppts = pts.get();
	if (options.fill) {
		cv::fillPoly(m_canvas, (const cv::Point**)&ppts, &npts, 1, cvColor(options.face_color));
	}
	cv::polylines(m_canvas, &ppts, &npts, 1, true, cvColor(options.edge_color), cvLineWidth(options.line_width));
}

void cvCanvas::DrawCircle(int x, int y, int radius, const DrawCircleOptions &options) {
	cv::circle(m_canvas, { x, y }, radius, cvColor(options.color), options.fill ? -1 : 1);
}

void cvCanvas::DrawText(int x, int y, const std::string &text, const DrawTextOptions &options) {
	// TODO: ignoring these options for now:
	// options.zorder
	// options.rotation
	// options.font_family

	int fontFace = cv::FONT_HERSHEY_DUPLEX;
	double fontScale = cvFontScale(options.font_size);
	int thickness = 1;
	int baseline;
	auto bbox_size = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

	switch (options.horizontal_alignment) {
	case kLeft:								 break;
	case kCenter: x -= bbox_size.width / 2;	 break;
	case kRight:  x -= bbox_size.width;		 break;
	default: assert(false);					 break;
	}
	switch (options.vertical_alignment) {
	case kTop:	  y += bbox_size.height;	 break;
	case kMiddle: y += bbox_size.height / 2; break;
	case kBottom:							 break;
	default: assert(false);					 break;
	}

	int padding = (int)options.bbox_padding;
	cv::Rect bbox_rect{ x - padding, y - bbox_size.height - padding,
		bbox_size.width + padding * 2, bbox_size.height + baseline + padding * 2 };

	if (!options.bbox_color.empty() || !options.edge_color.empty()) {
		if (options.edge_color.empty()) {
			cv::rectangle(m_canvas, bbox_rect, cvColor(options.bbox_color, options.bbox_alpha), -1);
		}
		else {
			if (!options.bbox_color.empty()) {
				cv::Rect inner_rect{ bbox_rect.x + 1, bbox_rect.y + 1, bbox_rect.width - 2, bbox_rect.height - 2 };
				cv::rectangle(m_canvas, inner_rect, cvColor(options.bbox_color, options.bbox_alpha), -1);
			}
			cv::rectangle(m_canvas, bbox_rect, cvColor(options.edge_color, options.bbox_alpha));
		}
	}

	cv::putText(m_canvas, text, { x, y }, fontFace, fontScale, cvColor(options.font_color), thickness);
}

void cvCanvas::DrawImage(const torch::Tensor &img) {
	assert(img.dim() == 3 && img.size(2) == 4);
	m_canvas = image_to_mat(img.to(torch::kUInt8));
}
