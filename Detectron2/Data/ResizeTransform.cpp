#include "Base.h"
#include "ResizeTransform.h"

#include <Detectron2/Utils/Utils.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ResizeTransform::ResizeTransform(int h, int w, int new_h, int new_w, Interp interp) :
	m_h(h), m_w(w), m_new_h(new_h), m_new_w(new_w), m_interp(interp) {
}

torch::Tensor ResizeTransform::apply_image(torch::Tensor img, Interp interp) {
	assert(img.size(0) == m_h && img.size(1) == m_w);
	assert(img.dim() <= 4);
	if (interp == kNone) { // doh' original code didn't do this when img.dtype() != torch::kUInt8
		interp = m_interp;
	}

	if (img.dtype() == torch::kUInt8) {
		cv::Mat mimg = image_to_mat(img);
		cv::resize(mimg, mimg, { m_new_w, m_new_h }, 0.0, 0.0, interp);
		img = image_to_tensor(mimg);
	}
	else {
		auto shape = torch::tensor(img.sizes());
		auto shape_4d = shape.index({ Slice(None, 2) }) + torch::tensor({ 1 }) * (4 - shape.size(0)) +
			shape.index({ Slice(2, None) });
		img = img.view(vectorize(shape_4d)).permute({ 2, 3, 0, 1 }); // hw(c) -> nchw

		auto options = nn::functional::InterpolateFuncOptions()
			.size(vector<int64_t>{ m_new_h, m_new_w }).align_corners(false);
		switch (interp) {
		case kNEAREST:  options.mode(torch::kNearest);  break;
		case kBILINEAR: options.mode(torch::kBilinear); break;
		case kBICUBIC:  options.mode(torch::kBicubic);  break;
		default: assert(false); break;
		}
		img = nn::functional::interpolate(img, options);
		shape.index_put_({ Slice(None, 2) }, torch::tensor({ m_new_h, m_new_w }));
		img = img.permute({ 2, 3, 0, 1 }).view(vectorize(shape)); // nchw -> hw(c)
	}
	return img;
}

torch::Tensor ResizeTransform::apply_coords(torch::Tensor coords) {
	coords.index_put_({ Colon, 0 }, coords.index({ Colon, 0 }) * ((float)m_new_w / m_w));
	coords.index_put_({ Colon, 1 }, coords.index({ Colon, 1 }) * ((float)m_new_h / m_h));
	return coords;
}

torch::Tensor ResizeTransform::apply_segmentation(torch::Tensor segmentation) {
	return apply_image(segmentation, kNEAREST);
}

std::shared_ptr<Transform> ResizeTransform::inverse() {
	return make_shared<ResizeTransform>(m_new_h, m_new_w, m_h, m_w, m_interp);
}
