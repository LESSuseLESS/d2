#include "Base.h"
#include "MaskOps.h"

#include <Detectron2/Utils/Utils.h>
#include <Detectron2/Data/Transform.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const int BYTES_PER_FLOAT = 4;
// TODO: This memory limit may be too much or too little. It would be better to
// determine it based on available resources.
const int GPU_MEM_LIMIT = 1024 * 1024 * 1024; // 1 GB memory limit

std::tuple<torch::Tensor, TensorVec> MaskOps::_do_paste_mask(torch::Tensor masks, torch::Tensor boxes,
	int img_h, int img_w, bool skip_empty) {
	// On GPU, paste all masks together (up to chunk size)
	// by using the entire image to sample the masks
	// Compared to pasting them one by one,
	// this has more operations but is faster on COCO-scale dataset.
	auto device = masks.device();
	int x0_int = 0;
	int y0_int = 0;
	int x1_int = img_w;
	int y1_int = img_h;
	if (skip_empty) {
		auto clamped = torch::clamp(boxes.min_values(0).floor().index({ { None, 2} }) - 1, 0).to(torch::kInt32);
		auto x0_int = clamped[0].item<int>();
		auto y0_int = clamped[1].item<int>();
		auto x1_int = torch::clamp(boxes.index({ Colon, 2 }).max().ceil() + 1, nullopt, img_w)
			.to(torch::kInt32).item<int>();
		auto y1_int = torch::clamp(boxes.index({ Colon, 3 }).max().ceil() + 1, nullopt, img_h)
			.to(torch::kInt32).item<int>();
	}
	auto splitted = torch::split(boxes, 1, 1);  // each is Nx1
	auto x0 = splitted[0];
	auto y0 = splitted[1];
	auto x1 = splitted[2];
	auto y1 = splitted[3];

	auto N = masks.size(0);

	auto img_y = torch::arange(y0_int, y1_int, dtype(torch::kFloat32).device(device)) + 0.5;
	auto img_x = torch::arange(x0_int, x1_int, dtype(torch::kFloat32).device(device)) + 0.5;
	img_y = (img_y - y0) / (y1 - y0) * 2 - 1;
	img_x = (img_x - x0) / (x1 - x0) * 2 - 1;
	// img_x, img_y have shapes (N, w), (N, h)

	auto gx = img_x.index({ Colon, None, Colon }).expand({ N, img_y.size(1), img_x.size(1) });
	auto gy = img_y.index({ Colon, Colon, None }).expand({ N, img_y.size(1), img_x.size(1) });
	auto grid = torch::stack({ gx, gy }, 3);

	auto img_masks = nn::functional::grid_sample(masks.to(torch::kFloat32), grid,
		nn::functional::GridSampleFuncOptions().align_corners(false));

	if (skip_empty) {
		return { img_masks.index({ Colon, 0 }),
			{ slice_range(y0_int, y1_int), slice_range(x0_int, x1_int) } };
	}
	return { img_masks.index({ Colon, 0 }), {} };
}

torch::Tensor MaskOps::paste_masks_in_image(torch::Tensor masks, torch::Tensor boxes, const ImageSize &image_shape,
	float threshold) {
	assert(masks.size(-1) == masks.size(-2)); // "Only square mask predictions are supported"
	auto N = masks.size(0);
	if (N == 0) {
		return masks.new_empty({ 0, image_shape.height, image_shape.width }, torch::kUInt8);
	}
	auto device = boxes.device();
	assert(boxes.size(0) == N);

	int img_h = image_shape.height;
	int img_w = image_shape.width;

	// The actual implementation split the input into chunks,
	// and paste them chunk by chunk.
	int num_chunks;
	if (device.type() == torch::kCPU) {
		// CPU is most efficient when they are pasted one by one with skip_empty=True
		// so that it performs minimal number of operations.
		num_chunks = N;
	}
	else {
		// GPU benefits from parallelism for larger chunks, but may have memory issue
		// int(img_h) because shape may be tensors in tracing
		num_chunks = int(ceil((float)N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT));
		assert(num_chunks <= N); // Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it
	}
	auto chunks = torch::chunk(torch::arange(N, device), num_chunks);

	auto img_masks = torch::zeros({ N, img_h, img_w },
		dtype(threshold >= 0 ? torch::kBool : torch::kUInt8).device(device));
	for (auto inds : chunks) {
		torch::Tensor masks_chunk;
		TensorVec spatial_inds;
		tie(masks_chunk, spatial_inds) = _do_paste_mask(
			masks.index({ inds, None, Colon, Colon }), boxes.index(inds), img_h, img_w, (device.type() == kCPU)
		);

		if (threshold >= 0) {
			masks_chunk = (masks_chunk >= threshold).to(torch::kBool);
		}
		else {
			// for visualization and debugging
			masks_chunk = (masks_chunk * 255).to(torch::kUInt8);
		}

		spatial_inds.insert(spatial_inds.begin(), inds);
		for (auto inds : spatial_inds) {
			img_masks.index_put_({ inds }, masks_chunk);
		}
	}
	return img_masks;
}

torch::Tensor MaskOps::paste_mask_in_image_old(torch::Tensor mask, torch::Tensor box, int img_h, int img_w,
	float threshold) {
	// Conversion from continuous box coordinates to discrete pixel coordinates
	// via truncation (cast to int32). This determines which pixels to paste the
	// mask onto.
	box = box.to(torch::kInt32);  // Continuous to discrete coordinate conversion
	// An example (1D) box with continuous coordinates (x0=0.7, x1=4.3) will map to
	// a discrete coordinates (x0=0, x1=4). Note that box is mapped to 5 = x1 - x0 + 1
	// pixels (not x1 - x0 pixels).
	int samples_w = box[2].item<int>() - box[0].item<int>() + 1;  // Number of pixel samples, *not* geometric width
	int samples_h = box[3].item<int>() - box[1].item<int>() + 1;  // Number of pixel samples, *not* geometric height

	// Resample the mask from it's original grid to the new samples_w x samples_h grid

	cv::Mat mmask = image_to_mat(mask);
	cv::resize(mmask, mmask, { samples_w, samples_h }, 0.0, 0.0, Transform::Interp::kBILINEAR);
	mask = image_to_tensor(mmask);
	if (threshold >= 0) {
		mask = (mask > threshold).to(torch::kUInt8);
	}
	else {
		// for visualization and debugging, we also
		// allow it to return an unmodified mask
		mask = (mask * 255).to(torch::kUInt8);
	}

	auto im_mask = torch::zeros({ img_h, img_w }, torch::kUInt8);
	auto x_0 = max(box[0].item<int>(), 0);
	auto x_1 = min(box[2].item<int>() + 1, img_w);
	auto y_0 = max(box[1].item<int>(), 0);
	auto y_1 = min(box[3].item<int>() + 1, img_h);

	im_mask.index_put_({ Slice(y_0, y_1), Slice(x_0, x_1) },
		mask.index({
			Slice(y_0 - box[1].item<int>(), y_1 - box[1].item<int>()),
			Slice(x_0 - box[0].item<int>(), x_1 - box[0].item<int>()) }));
	return im_mask;
}

std::tuple<torch::Tensor, float> MaskOps::pad_masks(torch::Tensor masks, int padding) {
	auto B = masks.size(0);
	auto M = masks.size(-1);
	auto pad2 = 2 * padding;
	auto scale = float(M + pad2) / M;
	auto padded_masks = masks.new_zeros((B, M + pad2, M + pad2));
	padded_masks.index_put_({ Colon, Slice(padding, -padding), Slice(padding, -padding) }, masks);
	return { padded_masks, scale };
}

torch::Tensor MaskOps::scale_boxes(torch::Tensor boxes, float scale) {
	auto w_half = (boxes.index({ Colon, 2 }) - boxes.index({ Colon, 0 })) * 0.5;
	auto h_half = (boxes.index({ Colon, 3 }) - boxes.index({ Colon, 1 })) * 0.5;
	auto x_c = (boxes.index({ Colon, 2 }) + boxes.index({ Colon, 0 })) * 0.5;
	auto y_c = (boxes.index({ Colon, 3 }) + boxes.index({ Colon, 1 })) * 0.5;

	w_half *= scale;
	h_half *= scale;

	auto scaled_boxes = torch::zeros_like(boxes);
	scaled_boxes.index_put_({ Colon, 0 }, x_c - w_half);
	scaled_boxes.index_put_({ Colon, 2 }, x_c + w_half);
	scaled_boxes.index_put_({ Colon, 1 }, y_c - h_half);
	scaled_boxes.index_put_({ Colon, 3 }, y_c + h_half);
	return scaled_boxes;
}
