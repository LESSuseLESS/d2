#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// assert in debug build; throw in release build
	void verify(bool expr);

	// string functions
	std::vector<std::string> tokenize(const std::string &input, char delimiter);
	std::string lower(const std::string &s);
	bool endswith(const std::string &s, const std::string &ending);
	std::string replace_all(const std::string &s, const std::string &src, const std::string &target);

	// image functions
	torch::Tensor mat_to_tensor(const cv::Mat &mat);
	cv::Mat image_to_mat(const torch::Tensor &t);
	torch::Tensor image_to_tensor(const cv::Mat &mat);
	torch::Tensor read_image(const std::string &pathname, const std::string &format = "");
}
