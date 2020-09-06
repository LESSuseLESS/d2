#include "Base.h"
#include "Utils.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Detectron2::verify(bool expr) {
	if (!expr) {
		throw std::exception("verify failed");
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// string functions

vector<string> Detectron2::tokenize(const string &input, char delimiter) {
	vector<string> s;
	size_t pos0 = 0;
	while (true) {
		size_t pos1 = input.find(delimiter, pos0);
		if (pos1 == string::npos) {
			s.push_back(input.substr(pos0));
			break;
		}
		s.push_back(input.substr(pos0, pos1 - pos0));
		pos0 = pos1 + 1;
	}
	assert(s.size() >= 1);
	return s;
}

std::string Detectron2::lower(const std::string &s) {
	std::string ret;
	ret.reserve(s.length());
	for (auto ch : s) {
		if (ch >= 'a' && ch < 'z') {
			ret += ch + ('A' - 'a');
		}
		else {
			ret += ch;
		}
	}
	return ret;
}

bool Detectron2::endswith(const std::string &s, const std::string &ending) {
	return s.length() >= ending.length() && s.substr(s.size() - ending.length()) == ending;
}

std::string Detectron2::replace_all(const std::string &s, const std::string &src, const std::string &target) {
	assert(!src.empty());

	string ret = s;
	auto pos = ret.find(src);
	while (pos != string::npos) {
		ret.replace(pos, src.length(), target);
		pos += target.length(); // so we don't run into infinite loop, if target.find(src) != string::npos
		pos = ret.find(src, pos);
	}
	return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// image functions

cv::Mat Detectron2::image_to_mat(const torch::Tensor &t) {
	int channel = 1;
	if (t.dim() != 2) {
		assert(t.dim() == 3);
		channel = t.size(2);
		assert(channel >= 1 && channel <= 4);
	}

	int type, size;
	if (t.dtype() == torch::kFloat32) {
		type = vector<int>{ CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4 }[channel - 1];
		size = sizeof(float);
	}
	else {
		assert(t.dtype() == torch::kUInt8);
		type = vector<int>{ CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4 }[channel - 1];
		size = sizeof(unsigned char);
	}

	cv::Mat ret((int)t.size(0), (int)t.size(1), type);
	::memcpy(ret.data, t.cpu().data_ptr(), t.numel() * size);
	return ret;
}

torch::Tensor Detectron2::mat_to_tensor(const cv::Mat &mat) {
	assert(mat.isContinuous());

	static map<int, std::tuple<torch::ScalarType, size_t, int>> types = {
		{ CV_32S,	{ torch::kInt32,	sizeof(int32_t), 1 }},
		{ CV_8U,	{ torch::kUInt8,	sizeof(char),	 1 }},
		{ CV_8UC3,	{ torch::kUInt8,	sizeof(char),	 3 }},
		{ CV_8UC4,	{ torch::kUInt8,	sizeof(char),	 4 }},
	};
	auto tt = mat.type();
	assert(types.find(tt) != types.end());
	torch::ScalarType type; size_t size; int channel;
	tie(type, size, channel) = types[tt];

	vector<int64_t> shape;
	for (int i = 0; i < mat.dims; i++) shape.push_back(mat.size[i]);
	if (channel > 1) {
		shape.push_back(channel);
	}
	int count = mat.size().area() * channel * size;

	auto p = malloc(count);
	memcpy(p, mat.data, count);
	return torch::from_blob(p, shape, torch::Deleter(free), type);
}

torch::Tensor Detectron2::image_to_tensor(const cv::Mat &mat) {
	assert(mat.isContinuous());
	assert(mat.type() == CV_8UC3 || mat.type() == CV_8UC4);
	assert(mat.dims == 2);

	int channel = mat.channels();
	int count = mat.size().area() * channel;
	auto p = malloc(count);
	memcpy(p, (unsigned char*)mat.data, count);
	return torch::from_blob(p, { mat.size[0], mat.size[1], channel }, torch::Deleter(free), torch::kUInt8);
}

torch::Tensor Detectron2::read_image(const std::string &pathname, const std::string &format) {
	// see data/detection_utils.py
	auto image = image_to_tensor(cv::imread(pathname));
	if (format == "BGR") {
		// flip channels if needed
		image = torch::flip(image, { -1 });
	}
	else if (format == "YUV-BT.601") {
		image = image / 255.0;

		// https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
		static torch::Tensor _M_RGB2YUV = torch::tensor(
			{ { 0.299, 0.587, 0.114 }, { -0.14713, -0.28886, 0.436 }, { 0.615, -0.51499, -0.10001 } });
		static torch::Tensor _M_YUV2RGB = torch::tensor(
			{ { 1.0, 0.0, 1.13983}, {1.0, -0.39465, -0.58060}, {1.0, 2.03211, 0.0} });

		image = torch::dot(image, _M_RGB2YUV.transpose(0, 1));
	}
	else {
		assert(format.empty());
	}

    return image;
}
