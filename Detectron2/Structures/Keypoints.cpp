#include "Base.h"
#include "Keypoints.h"

#include <Detectron2/Modules/Opeartors/NewEmptyTensorOp.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor Keypoints::interpolate(const torch::Tensor &input,
	const torch::nn::functional::InterpolateFuncOptions &options) {
	if (input.numel() > 0) {
		return torch::nn::functional::interpolate(input, options);
	}

	auto _check_size_scale_factor = [=](int dim) {
		assert(options.size().has_value() || options.scale_factor().has_value());
		assert(!options.size().has_value() || !options.scale_factor().has_value());
		if (options.scale_factor().has_value()) {
			assert(options.scale_factor()->size() == dim);
		}
	};

	auto _output_size = [=](int dim){
		_check_size_scale_factor(dim);
		if (options.size().has_value()) {
			return *options.size();
		}
		vector<int64_t> ret;
		for (int i = 0; i < dim; i++) {
			ret.push_back(input.size(i + 2) * (*options.scale_factor())[i]);
		}
		return ret;
	};
	vector<int64_t> output_shape = _output_size(2);
	return _NewEmptyTensorOp::apply(input,
		vector<int64_t>{ input.size(-2), input.size(-1), output_shape[0], output_shape[1] })[0];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Keypoints::Keypoints(const torch::Tensor &keypoints) {
	m_tensor = keypoints.to(dtype(torch::kFloat32).device(keypoints.device()));
	assert(m_tensor.dim() == 3 and m_tensor.size(2) == 3);
}

std::tuple<torch::Tensor, torch::Tensor> Keypoints::to_heatmap(torch::Tensor boxes, int heatmap_size) {
	return _keypoints_to_heatmap(m_tensor, boxes, heatmap_size);
}

Keypoints Keypoints::operator[](int64_t item) const {
	return m_tensor[item];
}

Keypoints Keypoints::operator[](ArrayRef<torch::indexing::TensorIndex> item) const {
	return m_tensor.index(item);
}

std::string Keypoints::toString() const {
	string s = "Keypoints(";
	s += FormatString("num_instances=%d)", len());
	return s;
}

std::tuple<torch::Tensor, torch::Tensor> Keypoints::_keypoints_to_heatmap(torch::Tensor keypoints,
	torch::Tensor rois, int heatmap_size) {
	if (rois.numel() == 0) {
		return { torch::tensor({}, torch::kInt64), torch::tensor({}, torch::kInt64) };
	}
	auto offset_x = rois.index({ Colon, 0 });
	auto offset_y = rois.index({ Colon, 1 });
	auto scale_x = heatmap_size / (rois.index({ Colon, 2 }) - rois.index({ Colon, 0 }));
	auto scale_y = heatmap_size / (rois.index({ Colon, 3 }) - rois.index({ Colon, 1 }));

	offset_x = offset_x.index({ Colon, None });
	offset_y = offset_y.index({ Colon, None });
	scale_x = scale_x.index({ Colon, None });
	scale_y = scale_y.index({ Colon, None });

	auto x = keypoints.index({ Ellipsis, 0 });
	auto y = keypoints.index({ Ellipsis, 1 });

	auto x_boundary_inds = x == rois.index({ Colon, 2 }).index({ Colon, None });
	auto y_boundary_inds = y == rois.index({ Colon, 3 }).index({ Colon, None });

	x = (x - offset_x) * scale_x;
	x = x.floor().to(torch::kInt64);
	y = (y - offset_y) * scale_y;
	y = y.floor().to(torch::kInt64);

	x.index_put_({ x_boundary_inds }, heatmap_size - 1);
	y.index_put_({ y_boundary_inds }, heatmap_size - 1);

	auto valid_loc = (x >= 0).bitwise_and(y >= 0).bitwise_and(x < heatmap_size).bitwise_and(y < heatmap_size);
	auto vis = keypoints.index({ Ellipsis, 2 }) > 0;
	auto valid = (valid_loc.bitwise_and(vis)).to(torch::kInt64);

	auto lin_ind = y * heatmap_size + x;
	auto heatmaps = lin_ind * valid;

	return { heatmaps, valid };
}

torch::Tensor Keypoints::heatmaps_to_keypoints(torch::Tensor maps, torch::Tensor rois) {
	torch::NoGradGuard guard;

	auto offset_x = rois.index({ Colon, 0 });
	auto offset_y = rois.index({ Colon, 1 });

	auto widths = (rois.index({ Colon, 2 }) - rois.index({ Colon, 0 })).clamp(1);
	auto heights = (rois.index({ Colon, 3 }) - rois.index({ Colon, 1 })).clamp(1);
	auto widths_ceil = widths.ceil();
	auto heights_ceil = heights.ceil();

	auto num_rois = maps.size(0);
	auto num_keypoints = maps.size(1);
	auto xy_preds = maps.new_zeros({ rois.size(0), num_keypoints, 4 });

	auto width_corrections = widths / widths_ceil;
	auto height_corrections = heights / heights_ceil;

	auto keypoints_idx = torch::arange(num_keypoints, maps.device());

	for (int i = 0; i < num_rois; i++) {
		vector<int64_t> outsize{ heights_ceil[i].item<int64_t>(), widths_ceil[i].item<int64_t>() };
		auto options = nn::functional::InterpolateFuncOptions()
			.size(outsize).mode(torch::kBicubic).align_corners(false);
		// doh' libtorch treat (1,...) vs (2,...) differently when doing maps.index({ Slice(i) })
		auto roi_map = interpolate(torch::stack({ maps[i] }), options).squeeze(0);  // #keypoints x H x W

		// softmax over the spatial region
		auto max_score = roi_map.view({ num_keypoints, -1 }).max_values(1);
		max_score = max_score.view({ num_keypoints, 1, 1 });
		auto tmp_full_resolution = (roi_map - max_score).exp_();
		auto tmp_pool_resolution = (maps[i] - max_score).exp_();
		// Produce scores over the region H x W, but normalize with POOL_H x POOL_W,
		// so that the scores of objects of different absolute sizes will be more comparable
		auto roi_map_scores = tmp_full_resolution / tmp_pool_resolution.sum({ 1, 2 }, true);

		auto w = roi_map.size(2);
		auto pos = roi_map.view({ num_keypoints, -1 }).argmax(1);

		auto x_int = pos % w;
		auto y_int = (pos - x_int).floor_divide(w);

		assert((roi_map_scores.index({ keypoints_idx, y_int, x_int })
				== roi_map_scores.view({ num_keypoints, -1 }).max_values(1)).all().item<bool>());

		auto x = (x_int.to(torch::kFloat32) + 0.5) * width_corrections[i];
		auto y = (y_int.to(torch::kFloat32) + 0.5) * height_corrections[i];

		xy_preds.index_put_({ i, Colon, 0 }, x + offset_x[i]);
		xy_preds.index_put_({ i, Colon, 1 }, y + offset_y[i]);
		xy_preds.index_put_({ i, Colon, 2 }, roi_map.index({ keypoints_idx, y_int, x_int }));
		xy_preds.index_put_({ i, Colon, 3 }, roi_map_scores.index({ keypoints_idx, y_int, x_int }));
	}
	return xy_preds;
}
