#include "Base.h"
#include "PostProcessing.h"

#include <Detectron2/Structures/Boxes.h>
#include <Detectron2/Structures/MaskOps.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

InstancesPtr PostProcessing::detector_postprocess(const InstancesPtr &results_,
	int output_height, int output_width, float mask_threshold) {
	auto scale_x = (float)output_width / results_->image_size().width;
	auto scale_y = (float)output_height / results_->image_size().height;
	InstancesPtr results(new Instances({ output_height, output_width }, results_->move_fields()));

	Tensor toutput_boxes;
	if (results->has("pred_boxes")) {
		toutput_boxes = results->getTensor("pred_boxes");
	}
	else if (results->has("proposal_boxes")) {
		toutput_boxes = results->getTensor("proposal_boxes");
	}
	auto output_boxes = Boxes::boxes(toutput_boxes);
	output_boxes->scale(scale_x, scale_y);
	output_boxes->clip(results->image_size());

	results = (*results)[output_boxes->nonempty()];

	if (results->has("pred_masks")) {
		retry_if_cuda_oom([&]() {
			results->set("pred_masks",
				MaskOps::paste_masks_in_image(
					results->getTensor("pred_masks").index({ Colon, 0, Colon, Colon }), // N, 1, M, M
					results->getTensor("pred_boxes"),
					results->image_size(),
					mask_threshold));
		});
	}
	if (results->has("pred_keypoints")) {
		Tensor t = results->getTensor("pred_keypoints");
		t.index_put_({ Colon, Colon, 0 }, t.index({ Colon, Colon, 0 }) * scale_x);
		t.index_put_({ Colon, Colon, 1 }, t.index({ Colon, Colon, 1 }) * scale_y);
		results->set("pred_keypoints", t); // this isn't necessary in theory
	}
	return results;
}

torch::Tensor PostProcessing::sem_seg_postprocess(torch::Tensor result, const ImageSize &img_size,
	int output_height, int output_width) {
	auto sliceImageSizes = vector<torch::indexing::TensorIndex>{
		Colon,
		Slice(None, img_size.height),
		Slice(None, img_size.width)
	};
	result = result.index(sliceImageSizes).expand({ 1, -1, -1, -1 });
	auto options = nn::functional::InterpolateFuncOptions()
		.size(vector<int64_t>{ output_height, output_width })
		.mode(torch::kBilinear)
		.align_corners(false);
	return nn::functional::interpolate(result, options)[0];;
}
