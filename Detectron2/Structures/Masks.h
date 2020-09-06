#pragma once

#include "Sequence.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	// Base class for BitMasks or PolygonMasks, so we can store in Instances with polymorphism.
	class Masks : public Sequence {
	public:
		virtual ~Masks() {}

		virtual torch::Tensor crop_and_resize(torch::Tensor boxes, int mask_size) = 0;
	};
}