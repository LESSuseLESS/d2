#pragma once

#include "TransformGen.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from data/transform/transform_gen.py

	/**
		Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
		If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
	*/
	class ResizeShortestEdge : public TransformGen {
	public:
		/**
			short_edge_length (list[int]): If ``sample_style=="range"``,
				a [min, max] interval from which to sample the shortest edge length.
				If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
			max_size (int): maximum allowed longest edge length.
			sample_style (str): either "range" or "choice".
		*/
		ResizeShortestEdge(int short_edge_length, int64_t max_size = INT64_MAX,
			const std::string &sample_style = "range", Transform::Interp interp = Transform::kBILINEAR);
		ResizeShortestEdge(const std::vector<int> &short_edge_length, int64_t max_size = INT64_MAX,
			const std::string &sample_style = "range", Transform::Interp interp = Transform::kBILINEAR);

		virtual std::shared_ptr<Transform> get_transform(torch::Tensor img) override;

	private:
		std::vector<int> m_short_edge_length;
		int64_t m_max_size;
		std::string m_sample_style;
		bool m_is_range;
		Transform::Interp m_interp;
	};
}
