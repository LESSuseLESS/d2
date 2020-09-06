#include "Base.h"
#include "ROIAlignRotated.h"

#include <Detectron2/detectron2/ROIAlignRotated/ROIAlignRotated.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ROIAlignRotatedImpl::ROIAlignRotatedImpl(const Size2D &output_size, float spatial_scale,
	int sampling_ratio) :
	m_output_size(output_size),
	m_spatial_scale(spatial_scale),
	m_sampling_ratio(sampling_ratio)
{
}

Tensor ROIAlignRotatedImpl::forward(const Tensor &input, const Tensor &rois) {
	assert(rois.dim() == 2 and rois.size(1) == 6);
	return detectron2::ROIAlignRotated_forward(input, rois, m_spatial_scale,
		m_output_size.height, m_output_size.width, m_sampling_ratio);
}

std::string ROIAlignRotatedImpl::toString() const {
	std::string tmpstr = "ROIAlign(";
	tmpstr += "output_size=(" + torch::str(m_output_size.height) + ", " + torch::str(m_output_size.width) + ")";
	tmpstr += ", spatial_scale=" + torch::str(m_spatial_scale);
	tmpstr += ", sampling_ratio=" + torch::str(m_sampling_ratio);
	tmpstr += ")";
	return tmpstr;
}
