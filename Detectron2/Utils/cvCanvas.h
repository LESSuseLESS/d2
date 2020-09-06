#pragma once

#include "Canvas.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class cvCanvas : public Canvas {
	public:
		cvCanvas(int height, int width);

		// implementing Canvas
		virtual std::tuple<torch::Tensor, int, int> SaveToTensor() override;
		virtual void DrawLine2D(const std::vector<int> &x_data, const std::vector<int> &y_data,
			const DrawLine2DOptions &options) override;
		virtual void DrawRectangle(int x, int y, int width, int height, const DrawRectangleOptions &options) override;
		virtual void DrawPolygon(const torch::Tensor &segment, const DrawPolygonOptions &options) override;
		virtual void DrawCircle(int x, int y, int radius, const DrawCircleOptions &options) override;
		virtual void DrawText(int x, int y, const std::string &text, const DrawTextOptions &options) override;
		virtual void DrawImage(const torch::Tensor &img) override;

	protected:
		cv::Mat m_canvas;

		static cv::Scalar cvColor(const std::vector<float> &c);
		static cv::Scalar cvColor(const Color3 &c, float alpha);
		static int cvLineWidth(float line_width);
		static int cvLineType(LineStyle line_style);
		static double cvFontScale(float font_size);
	};
}