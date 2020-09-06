#include "Base.h"
#include "ROIAlign.h"

#include <Detectron2/detectron2/ROIAlign/ROIAlign.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ROIAlignImpl::ROIAlignImpl(const Size2D &output_size, float spatial_scale, int sampling_ratio, bool aligned) :
	m_output_size(output_size),
	m_spatial_scale(spatial_scale),
	m_sampling_ratio(sampling_ratio),
	m_aligned(aligned)
{
}

Tensor ROIAlignImpl::forward(const Tensor &input, const Tensor &rois) {
	assert(rois.dim() == 2 and rois.size(1) == 5);
	return detectron2::ROIAlign_forward(input, rois, m_spatial_scale, m_output_size.height, m_output_size.width,
		m_sampling_ratio, m_aligned);
}

std::string ROIAlignImpl::toString() const {
	std::string tmpstr = "ROIAlign(";
	tmpstr += "output_size=(" + torch::str(m_output_size.height) + ", " + torch::str(m_output_size.width) + ")";
	tmpstr += ", spatial_scale=" + torch::str(m_spatial_scale);
	tmpstr += ", sampling_ratio=" + torch::str(m_sampling_ratio);
	tmpstr += ", aligned=" + torch::str(m_aligned);
	tmpstr += ")";
	return tmpstr;
}
