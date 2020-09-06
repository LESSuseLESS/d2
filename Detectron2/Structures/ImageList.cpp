#include "Base.h"
#include "ImageList.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ImageList ImageList::from_tensors(const TensorVec &tensors, int size_divisibility, double pad_value) {
	assert(!tensors.empty());

	std::vector<ImageSize> image_sizes;
	{
		auto GetImageSize = [=](int index) -> ImageSize {
			auto sizes = tensors[index].sizes();
			auto dim = sizes.size();
			assert(dim > 1);
			auto h = sizes[dim - 2];
			auto w = sizes[dim - 1];
			return { (int)h, (int)w };
		};
		auto GetRemaining = [=](int index) -> IntArrayRef {
			auto sizes = tensors[index].sizes();
			int count = sizes.size() - 3;
			if (count < 0) count = 0;
			return sizes.slice(1, count);
		};
		auto remaining0 = GetRemaining(0);
		image_sizes.reserve(tensors.size());
		for (int i = 0; i < tensors.size(); i++) {
			assert(GetRemaining(i) == remaining0);
			image_sizes.push_back(GetImageSize(i));
		}
	}

	// per dimension maximum (H, W) or (C_1, ..., C_K, H, W) where K >= 1 among all tensors
	TensorVec dims;
	dims.reserve(tensors.size());
	for (auto &t : tensors) {
		dims.push_back(torch::tensor(t.sizes()));
	}
	// In tracing mode, x.shape[i] is Tensor, and should not be converted
	// to int: this will cause the traced graph to have hard-coded shapes.
	// Instead we should make max_size a Tensor that depends on these tensors.
	// Using torch.stack twice seems to be the best way to convert
	// list[list[ScalarTensor]] to a Tensor
	auto max_size = torch::stack(dims).max_values(0);

	if (size_divisibility > 0) {
		int stride = size_divisibility;
		// the last two dims are H,W, both subject to divisibility requirement
		max_size = torch::cat({ max_size.index({ Slice(None, -2) }),
			(max_size.index({ Slice(-2, None) }) + (stride - 1)).floor_divide(stride) * stride });
	}

	// max_size can be a tensor in tracing mode, therefore use tuple()
	assert(max_size.dtype() == torch::kInt64);
	vector<int64_t> batch_shape;
	batch_shape.reserve(max_size.size(0) + 1);
	batch_shape.push_back(tensors.size());
	for (int i = 0; i < max_size.size(0); i++) {
		batch_shape.push_back(max_size[i].item<int64_t>());
	}

	Tensor batched_imgs;
	if (tensors.size() == 1) {
		// This seems slightly (2%) faster.
		// TODO: check whether it's faster for multiple images as well
		auto image_size = image_sizes[0];
		vector<int64_t> padding_size{
			0, batch_shape[batch_shape.size() - 1] - image_size.width,
			0, batch_shape[batch_shape.size() - 2] - image_size.height
		};

		if (all_vec<int64_t>(padding_size, [](int64_t x){ return x == 0; })) {
			// https://github.com/pytorch/pytorch/issues/31734
			batched_imgs = tensors[0].unsqueeze(0);
		}
		else {
			auto padded = nn::functional::pad(tensors[0],
				nn::functional::PadFuncOptions(padding_size).value(pad_value));
			batched_imgs = padded.unsqueeze_(0);
		}
	}
	else {
		batched_imgs = tensors[0].new_full(batch_shape, pad_value);
		for (int i = 0; i < tensors.size(); i++) {
			auto &img = tensors[i];
			auto img_sizes = img.sizes();
			auto pad_img = batched_imgs[i];
			auto pad_img_sizes = pad_img.sizes();

			vector<int64_t> sizes;
			sizes.reserve(pad_img_sizes.size());
			for (int i = 0; i < pad_img_sizes.size() - 2; i++) {
				sizes.push_back(pad_img_sizes[i]);
			}
			auto dim = img_sizes.size();
			sizes.push_back(img_sizes[dim - 2]);
			sizes.push_back(img_sizes[dim - 1]);

			pad_img.view(sizes).copy_(img);
		}
	}
	return ImageList(batched_imgs.contiguous(), image_sizes);
}

torch::Tensor ImageList::get(int64_t idx) {
	assert(idx >= 0 && idx < m_image_sizes.size());
	auto size = m_image_sizes[idx];
	return m_tensor.index({ idx, Ellipsis,
		Slice(None, size.height), Slice(None, size.width) }); // type: ignore
}
