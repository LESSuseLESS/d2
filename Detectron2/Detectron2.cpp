#include "Base.h"
#include "Detectron2.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool Detectron2::cudaEnabled() {
	return torch::cuda::is_available();
}

void Detectron2::retry_if_cuda_oom(std::function<void()> func) {
	//~! we're not doing any retry yet
	func();
}

int Detectron2::IntLog2(int exp) {
	int n = 0;
	while (exp > 1) {
		assert(exp % 2 == 0);
		exp /= 2;
		n++;
	}
	assert(exp == 1);
	return n;
}

int Detectron2::IntExp2(int n) {
	int result = 1;
	for (int i = 0; i < n; i++) {
		result *= 2;
	}
	return result;
}

std::string Detectron2::FormatString(const char *fmt, int d) {
	char buf[256];
	snprintf(buf, sizeof(buf), fmt, d);
	return buf;
}

std::string Detectron2::FormatString(const char *fmt, double f) {
	char buf[256];
	snprintf(buf, sizeof(buf), fmt, f);
	return buf;
}

torch::Tensor Detectron2::slice_range(int64_t start, int64_t end, int64_t step) {
	vector<int64_t> range;
	range.reserve((end - start) / step + 1);
	for (int64_t i = start; i < end; i += step) {
		range.push_back(i);
	}
	return torch::tensor(range);
}

std::vector<int64_t> Detectron2::vectorize(const torch::Tensor &t) {
	assert(t.dim() == 1);
	vector<int64_t> ret;
	ret.reserve(t.numel());
	for (int i = 0; i < t.numel(); i++) {
		ret.push_back(t[i].item<int64_t>());
	}
	return ret;
}

torch::Tensor Detectron2::tapply(const torch::Tensor &src, function<torch::Tensor(torch::Tensor)> fx) {
	TensorVec ret;
	int count = src.size(0);
	ret.reserve(count);
	for (int i = 0; i < count; i++) {
		ret.push_back(fx(src[i]));
	}
	return torch::cat(ret);
}
