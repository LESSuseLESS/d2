#pragma once

#include <Detectron2/Structures/ShapeSpec.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/backbone.py

	// Abstract base class for network backbones.
	class BackboneImpl : public torch::nn::Module {
	public:
		virtual ~BackboneImpl() {}

		const ShapeSpec::Map &output_shapes() const {
			return m_output_shapes;
		}

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) = 0;

		/**
			Subclasses must override this method, but adhere to the same return type.

			Returns:
				dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
		*/
		virtual TensorMap forward(torch::Tensor x) = 0;

		/**
			Some backbones require the input height and width to be divisible by a
			specific integer. This is typically true for encoder / decoder type networks
			with lateral connection (e.g., FPN) for which feature maps need to match
			dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
			input size divisibility is required.
		*/
		virtual int size_divisibility() {
			return 0;
		}

	protected:
		ShapeSpec::Map m_output_shapes;
	};
	TORCH_MODULE(Backbone);
}