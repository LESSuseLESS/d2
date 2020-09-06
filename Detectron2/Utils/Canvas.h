#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	class Canvas {
	public:
		enum Alignment {
			kLeft,
			kCenter,
			kRight,

			kTop,
			kMiddle,
			kBottom,
		};

		// https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
		enum LineStyle {
			kSolid,		// '-'
			kDotted,	// '.'
			kDashed,	// '--'
			kDashDot,	// '-.'
		};

		// RGB or RGBA
		using Color3 = std::vector<float>;
		using Color4 = std::vector<float>;

	public:
		virtual ~Canvas() {}

		// buffer, width, height, alpha
		virtual std::tuple<torch::Tensor, int, int> SaveToTensor() = 0;

		struct DrawLine2DOptions {
			float line_width;
			Color3 color;
			LineStyle line_style;
		};
		virtual void DrawLine2D(const std::vector<int> &x_data, const std::vector<int> &y_data,
			const DrawLine2DOptions &options) = 0;

		struct DrawRectangleOptions {
			bool fill = false;
			Color3 edge_color;
			float line_width;
			float alpha;
			LineStyle line_style;
		};
		virtual void DrawRectangle(int x, int y, int width, int height, const DrawRectangleOptions &options) = 0;

		struct DrawPolygonOptions {
			bool fill = false;
			Color4 face_color;
			Color4 edge_color;
			float line_width;
		};
		virtual void DrawPolygon(const torch::Tensor &segment, const DrawPolygonOptions &options) = 0;

		struct DrawCircleOptions {
			bool fill = false;
			Color3 color;
		};
		virtual void DrawCircle(int x, int y, int radius, const DrawCircleOptions &options) = 0;

		struct DrawTextOptions {
			float font_size = 8;
			Color3 font_color;
			const char *font_family = "arial";
			Color3 bbox_color;
			float bbox_alpha = 1.0;
			float bbox_padding = 0.0;
			Color3 edge_color;
			Alignment vertical_alignment = kTop;
			Alignment horizontal_alignment = kLeft;
			int zorder = 0;
			float rotation = 0.0f;
		};
		virtual void DrawText(int x, int y, const std::string &text, const DrawTextOptions &options) = 0;

		// img in torch::kFloat32 with alpha
		virtual void DrawImage(const torch::Tensor &img) = 0;
	};
}
