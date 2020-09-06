#include "Base.h"
#include "AsyncPredictor.h"

#include "DefaultPredictor.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

AsyncPredictor::AsyncPredictor(const CfgNode &cfg, int num_gpus) : m_put_idx(0), m_get_idx(0) {
	auto _PredictWorker = [&](YAML::Node cfg) {
		DefaultPredictor predictor(cfg); // doh' this cloning is unnecessary
		while (true) {
			int idx; torch::Tensor data;
			m_task_queue_mutex.lock();
			tie(idx, data) = m_task_queue.front();
			if (idx >= 0) {
				m_task_queue.pop_front();
			}
			m_task_queue_mutex.unlock();

			if (idx < 0) break;
			auto result = predictor.predict(data);
			m_result_queue_mutex.lock();
			m_result_queue.push_back({ idx, result });
			m_result_queue_mutex.unlock();
		}
	};

	int num_workers = max(num_gpus, 1);
	for (int gpuid = 0; gpuid < num_workers; gpuid++) {
		CfgNode cloned(cfg.clone());
		cloned.defrost();
		if (num_gpus > 0) {
			cloned["MODEL.DEVICE"] = FormatString("cuda:%d", gpuid);
		}
		else {
			cloned["MODEL.DEVICE"] = "cpu";
		}
		m_procs.push_back(make_shared<thread>(_PredictWorker, cfg.node())); // .node() is safe, but not CfgNode
	}
}

void AsyncPredictor::put(torch::Tensor image) {
	m_task_queue_mutex.lock();
	m_task_queue.push_back({ ++m_put_idx, image });
	m_task_queue_mutex.unlock();
}

InstancesPtr AsyncPredictor::get() {
	++m_get_idx; // the index needed for this request
	m_result_rank_mutex.lock();
	if (!m_result_rank.empty() && std::get<0>(m_result_rank.front()) == m_get_idx) {
		auto res = std::get<1>(m_result_rank.front());
		m_result_rank.pop_front();
		m_result_rank_mutex.unlock();
		return res;
	}
	m_result_rank_mutex.unlock();

	while (true) {
		// make sure the results are returned in the correct order
		int idx; InstancesPtr res;
		m_result_queue_mutex.lock();
		tie(idx, res) = m_result_queue.front();
		m_result_queue.pop_front();
		m_result_queue_mutex.unlock();
		if (idx == m_get_idx) {
			return res;
		}
		m_result_rank_mutex.lock();
		for (auto iter = m_result_rank.begin(); iter != m_result_rank.end(); ++iter) {
			auto rank = std::get<0>(*iter);
			if (rank > idx) {
				m_result_rank.insert(iter, { idx, res });
			}
		}
		m_result_rank_mutex.unlock();
	}

	assert(false);
	return nullptr;
}

void AsyncPredictor::shutdown() {
	m_task_queue_mutex.lock();
	m_task_queue.push_back({ -1, Tensor() });
	m_task_queue_mutex.unlock();

	for (auto t : m_procs) {
		t->join();
	}
}
