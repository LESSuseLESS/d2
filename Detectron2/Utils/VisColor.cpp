#include "Base.h"
#include "VisColor.h"

using namespace std;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// RGB:
static float _COLORS[]{
		0.000, 0.447, 0.741,
		0.850, 0.325, 0.098,
		0.929, 0.694, 0.125,
		0.494, 0.184, 0.556,
		0.466, 0.674, 0.188,
		0.301, 0.745, 0.933,
		0.635, 0.078, 0.184,
		0.300, 0.300, 0.300,
		0.600, 0.600, 0.600,
		1.000, 0.000, 0.000,
		1.000, 0.500, 0.000,
		0.749, 0.749, 0.000,
		0.000, 1.000, 0.000,
		0.000, 0.000, 1.000,
		0.667, 0.000, 1.000,
		0.333, 0.333, 0.000,
		0.333, 0.667, 0.000,
		0.333, 1.000, 0.000,
		0.667, 0.333, 0.000,
		0.667, 0.667, 0.000,
		0.667, 1.000, 0.000,
		1.000, 0.333, 0.000,
		1.000, 0.667, 0.000,
		1.000, 1.000, 0.000,
		0.000, 0.333, 0.500,
		0.000, 0.667, 0.500,
		0.000, 1.000, 0.500,
		0.333, 0.000, 0.500,
		0.333, 0.333, 0.500,
		0.333, 0.667, 0.500,
		0.333, 1.000, 0.500,
		0.667, 0.000, 0.500,
		0.667, 0.333, 0.500,
		0.667, 0.667, 0.500,
		0.667, 1.000, 0.500,
		1.000, 0.000, 0.500,
		1.000, 0.333, 0.500,
		1.000, 0.667, 0.500,
		1.000, 1.000, 0.500,
		0.000, 0.333, 1.000,
		0.000, 0.667, 1.000,
		0.000, 1.000, 1.000,
		0.333, 0.000, 1.000,
		0.333, 0.333, 1.000,
		0.333, 0.667, 1.000,
		0.333, 1.000, 1.000,
		0.667, 0.000, 1.000,
		0.667, 0.333, 1.000,
		0.667, 0.667, 1.000,
		0.667, 1.000, 1.000,
		1.000, 0.000, 1.000,
		1.000, 0.333, 1.000,
		1.000, 0.667, 1.000,
		0.333, 0.000, 0.000,
		0.500, 0.000, 0.000,
		0.667, 0.000, 0.000,
		0.833, 0.000, 0.000,
		1.000, 0.000, 0.000,
		0.000, 0.167, 0.000,
		0.000, 0.333, 0.000,
		0.000, 0.500, 0.000,
		0.000, 0.667, 0.000,
		0.000, 0.833, 0.000,
		0.000, 1.000, 0.000,
		0.000, 0.000, 0.167,
		0.000, 0.000, 0.333,
		0.000, 0.000, 0.500,
		0.000, 0.000, 0.667,
		0.000, 0.000, 0.833,
		0.000, 0.000, 1.000,
		0.000, 0.000, 0.000,
		0.143, 0.143, 0.143,
		0.857, 0.857, 0.857,
		1.000, 1.000, 1.000
};

VisColor Detectron2::color_normalize(const VisColor &color) {
	VisColor ret;
	for (auto x : color) {
		ret.push_back(x / 255);
	}
	return ret;
}

VisColor Detectron2::color_denormalize(const VisColor &color) {
	VisColor ret;
	for (auto x : color) {
		ret.push_back(x * 255);
	}
	return ret;
}

VisColor Detectron2::color_at_least(const VisColor &color, float minimum) {
	VisColor ret;
	for (auto x : color) {
		ret.push_back(x < minimum ? minimum : x);
	}
	return ret;
}

VisColor Detectron2::color_from_tensor(const torch::Tensor &t) {
	return {
		t[0].item<float>(),
		t[1].item<float>(),
		t[2].item<float>()
	};
}

VisColor Detectron2::color_random(bool rgb, int maximum) {
	int idx = std::rand() % ((sizeof(_COLORS) / sizeof(_COLORS[0])) / 3);
	if (rgb) {
		return {
			_COLORS[idx + 0] * maximum,
			_COLORS[idx + 1] * maximum,
			_COLORS[idx + 2] * maximum
		};
	} else {
		return {
			_COLORS[idx + 2] * maximum,
			_COLORS[idx + 1] * maximum,
			_COLORS[idx + 0] * maximum
		};
	}
}

VisColor Detectron2::color_jitter(const VisColor &color) {
	auto vec = torch::rand(3);
	// better to do it in another color space
	vec = vec / torch::norm(vec) * 0.5;
	auto res = (vec + torch::tensor(color)).clamp(0, 1);
	return color_from_tensor(res);
}

VisColor Detectron2::color_brightness(const VisColor &color, float brightness_factor) {
	assert(brightness_factor >= -1.0 && brightness_factor <= 1.0);
	auto ret = color;
	for (int i = 0; i < 3; i++) {
		ret[i] *= (1.0f + brightness_factor);
		ret[i] = clip<float>(ret[i], 0.0, 1.0);
	}
	return ret;
}
