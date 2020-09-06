#pragma once

#include <Detectron2/Import/ModelImporter.h>
#include <Detectron2/Utils/CfgNode.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// data structures

	using TensorMap = std::unordered_map<std::string, torch::Tensor>;
	using TensorMapList = std::vector<std::shared_ptr<TensorMap>>;
	using TensorVec = std::vector<torch::Tensor>;
	using BoxesList = TensorVec;

	using ModulePtr = std::shared_ptr<torch::nn::Module>;

	struct Size2D {
		int height;
		int width;
	};
	using ImageSize = Size2D;	// height, width

	struct Position2D {
		int x;
		int y;
	};
	using Pos = Position2D;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// basics

	bool cudaEnabled();
	void retry_if_cuda_oom(std::function<void()> func);

	// utils/comm.py
	inline int get_world_size() { return 1; }

	int IntLog2(int exp);
	int IntExp2(int n);

	std::string FormatString(const char *fmt, int d);
	std::string FormatString(const char *fmt, double f);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Tensor: .tolist()

	const torch::indexing::TensorIndex Colon = torch::indexing::Slice();
	const torch::nullopt_t None = torch::indexing::None;
	const torch::indexing::EllipsisIndexType Ellipsis = torch::indexing::Ellipsis;
	using Slice = torch::indexing::Slice;

	torch::Tensor slice_range(int64_t start, int64_t end, int64_t step = 1);

	inline torch::Tensor tolist(const torch::Tensor &t) {
		return t;
	}

	template<typename T>
	T clip(T t, T t0, T t1) {
		if (t < t0) return t0;
		if (t > t1) return t1;
		return t;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// vector and comprehension

	std::vector<int64_t> vectorize(const torch::Tensor &t);

	template<typename T>
	void vec_select(std::vector<T> &dest, const std::vector<T> &src, const torch::Tensor &idxs) {
		int count = idxs.size(0);
		dest.reserve(count);
		for (int i = 0; i < count; i++) {
			auto index = idxs[i].item<int64_t>();
			assert(index >= 0 && index < src.size());
			dest.push_back(src[index]);
		}
	}

	template<typename T>
	std::vector<T> vapply(const std::vector<T> &src, std::function<T(T)> fx) {
		std::vector<T> ret;
		ret.reserve(src.size());
		for (auto &x : src) {
			ret.push_back(fx(x));
		}
		return ret;
	}
	torch::Tensor tapply(const torch::Tensor &src, std::function<torch::Tensor(torch::Tensor)> fx);

	template<typename Target, typename Source>
	std::vector<Target> vapply(const std::vector<Source> &src, std::function<Target(Source)> fx) {
		std::vector<Target> ret;
		ret.reserve(src.size());
		for (auto &x : src) {
			ret.push_back(fx(x));
		}
		return ret;
	}

	template<typename T, typename Iter>
	bool all(const T &collection, std::function<bool(Iter)> test) {
		for (auto x : collection) {
			if (!test(x)) {
				return false;
			}
		}
		return true;
	}
	template<typename T>
	bool all_vec(const std::vector<T> &collection, std::function<bool(T)> test) {
		return all<std::vector<T>, T>(collection, test);
	}
}
