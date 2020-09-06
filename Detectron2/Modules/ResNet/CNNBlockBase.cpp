#include "Base.h"
#include "CNNBlockBase.h"

#include <Detectron2/Modules/BatchNorm/FrozenBatchNorm2d.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CNNBlockBaseImpl::CNNBlockBaseImpl(int in_channels, int out_channels, int stride) :
	m_in_channels(in_channels), m_out_channels(out_channels), m_stride(stride) {
}

void CNNBlockBaseImpl::freeze() {
	for (auto p : parameters()) {
		p.set_requires_grad(false);
	}
	auto self = shared_from_this();
	FrozenBatchNorm2dImpl::convert_frozen_batchnorm(self);
}
