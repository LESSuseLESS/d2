#pragma once

#include "Predictor.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from demo/predictor.py

    /**
		A predictor that runs the model asynchronously, possibly on >1 GPUs.
		Because rendering the visualization takes considerably amount of time,
		this helps improve throughput when rendering videos.
	*/
	class AsyncPredictor : public Predictor {
	public:
		/**
			cfg (CfgNode):
			num_gpus (int): if 0, will run on CPU
		*/
		AsyncPredictor(const CfgNode &cfg, int num_gpus = 1);

		int64_t len() const { return m_put_idx - m_get_idx; }
		int default_buffer_size() const { return m_procs.size() * 5; }

		void put(torch::Tensor image);
		InstancesPtr get();
		InstancesPtr operator()(torch::Tensor image) { return predict(image); }
		virtual InstancesPtr predict(torch::Tensor original_image) override {
			put(original_image);
			return get();
		}

		void shutdown();

	private:
		std::mutex m_task_queue_mutex;
		std::list<std::tuple<int, torch::Tensor>> m_task_queue;
		std::mutex m_result_queue_mutex;
		std::list<std::tuple<int, InstancesPtr>> m_result_queue;
		std::vector<std::shared_ptr<std::thread>> m_procs;

		int m_put_idx;
		int m_get_idx;
		std::mutex m_result_rank_mutex;
		std::list<std::tuple<int, InstancesPtr>> m_result_rank;
	};
}
