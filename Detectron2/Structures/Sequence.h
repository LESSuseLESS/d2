#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class Sequence;
	using SequencePtr = std::shared_ptr<Sequence>;
	using SequencePtrMap = std::unordered_map<std::string, SequencePtr>;

	class Sequence {
	public:
		virtual ~Sequence() {}

		virtual int size() const = 0;
		virtual std::string toString() const = 0;
		virtual SequencePtr slice(int64_t start, int64_t end) const = 0;
		virtual SequencePtr index(int item) const { return slice(item, item + 1); }
		virtual SequencePtr index(torch::Tensor item) const = 0;
		virtual SequencePtr cat(const std::vector<SequencePtr> &seqs, int total) const = 0;

		SequencePtr operator[](int item) const { return index(item); }
		SequencePtr operator[](torch::Tensor item) const { return index(item); }
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class SequenceTensor : public Sequence {
	public:
		SequenceTensor(torch::Tensor t) : m_data(t) {}

		torch::Tensor data() const { return m_data; }

		virtual int size() const override {
			return m_data.size(0);
		}
		virtual std::string toString() const override {
			return m_data.toString();
		}
		virtual SequencePtr slice(int64_t start, int64_t end) const override;
		virtual SequencePtr index(torch::Tensor item) const override;
		virtual SequencePtr cat(const std::vector<SequencePtr> &seqs, int total) const override;

	private:
		torch::Tensor m_data;
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename T>
	class SequenceVec : public Sequence {
	public:
		const std::vector<T> &data() const { return m_data; }
		std::vector<T> &data() { return m_data; }

		virtual int size() const override {
			return m_data.size();
		}
		virtual std::string toString() const override {
			return FormatString("[%d items]", (int)m_data.size());
		}
		virtual SequencePtr slice(int64_t start, int64_t end) const override {
			assert(start >= 0 && start < m_data.size());
			assert(end >= 0 && end <= m_data.size());

			int count = (end - start - 1);
			std::shared_ptr<SequenceVec<T>> sliced(new SequenceVec<T>());
			if (count > 0) {
				sliced->m_data.reserve(count);
				for (int i = start; i < end; i++) {
					sliced->m_data.push_back(m_data[i]);
				}
			}
			return sliced;
		}
		virtual SequencePtr index(torch::Tensor item) const override {
			assert(item.dim() == 1 && item.dtype() == torch::kInt64);
			std::shared_ptr<SequenceVec<T>> selected(new SequenceVec<T>());
			int count = item.size(0);
			selected->m_data.reserve(count);
			for (int i = 0; i < count; i++) {
				auto index = item[i].item<int64_t>();
				assert(index >= 0 && index < m_data.size());
				selected->m_data.push_back(m_data[index]);
			}
			return selected;
		}
		virtual SequencePtr cat(const std::vector<SequencePtr> &seqs, int total) const override {
			std::shared_ptr<SequenceVec<T>> aggregated(new SequenceVec<T>());
			auto &res = aggregated->m_data;
			res.reserve(total);
			for (int i = 0; i < seqs.size(); i++) {
				auto &data = std::dynamic_pointer_cast<SequenceVec<T>>(seqs[i])->m_data;
				res.insert(res.end(), data.begin(), data.end());
			}
			assert(res.size() == total);
			return aggregated;
		}

	private:
		std::vector<T> m_data;
	};
}
