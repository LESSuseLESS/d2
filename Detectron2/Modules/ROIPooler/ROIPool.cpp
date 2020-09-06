#include "Base.h"
#include "RoIPool.h"

#include <Detectron2/detectron2/ROIPool/ROIPool.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// converted from https://github.com/pytorch/vision torchvision/ops/_utils.py

// Efficient version of torch.cat that avoids a copy if there is only a single element in a list
static torch::Tensor _cat(const TensorVec &tensors, int dim = 0) {
	// TODO add back the assert
	// assert isinstance(tensors, (list, tuple))
	if (tensors.size() == 1) {
		return tensors[0];
	}
	return torch::cat(tensors, dim);
}

static torch::Tensor convert_boxes_to_roi_format(const BoxesList &boxes) {
	auto concat_boxes = _cat(boxes, 0);
	TensorVec temp;
	temp.reserve(boxes.size());
	for (int i = 0; i < boxes.size(); i++) {
		auto &b = boxes[i];
		temp.push_back(torch::full_like(b.index({ Colon, Slice(None, 1) }), i));
	}
	auto ids = _cat(temp, 0);
	auto rois = torch::cat({ ids, concat_boxes }, 1);
	return rois;
}

static void check_roi_boxes_shape(const BoxesList &boxes) {
	for (auto &_tensor : boxes) {
		assert(_tensor.size(1) == 4); // The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]
	}
}

static void check_roi_boxes_shape(const torch::Tensor &boxes) {
	assert(boxes.size(1) == 5); // The boxes tensor shape is not correct as Tensor[K, 5]
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor Detectron2::roi_pool(const torch::Tensor &input, const BoxesList &boxes, const Size2D &output_size,
	float spatial_scale) {
	check_roi_boxes_shape(boxes);
	auto rois = convert_boxes_to_roi_format(boxes);
	auto output = get<0>(detectron2::roi_pool(input, rois, spatial_scale, output_size.height, output_size.width));
	return output;
}

torch::Tensor Detectron2::roi_pool(const torch::Tensor &input, const torch::Tensor &boxes, const Size2D &output_size,
	float spatial_scale) {
	check_roi_boxes_shape(boxes);
	auto rois = boxes;
	auto output = get<0>(detectron2::roi_pool(input, rois, spatial_scale, output_size.height, output_size.width));
	return output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RoIPoolImpl::RoIPoolImpl(const Size2D &output_size, float spatial_scale) :
	m_output_size(output_size),
	m_spatial_scale(spatial_scale)
{
}

torch::Tensor RoIPoolImpl::forward(const torch::Tensor &input, const torch::Tensor &rois) {
	return roi_pool(input, rois, m_output_size, m_spatial_scale);
}

std::string RoIPoolImpl::toString() const {
	std::string tmpstr = "RoIPool(";
	tmpstr += "output_size=(" + torch::str(m_output_size.height) + ", " + torch::str(m_output_size.width) + ")";
	tmpstr += ", spatial_scale=" + torch::str(m_spatial_scale);
	tmpstr += ")";
	return tmpstr;
}
