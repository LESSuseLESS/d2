#include "Base.h"
#include "Instances.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TensorVec InstancesList::getTensorVec(const std::string &name) const {
	return vapply<torch::Tensor, InstancesPtr>(*this,
		[=](const InstancesPtr &instance){ return instance->getTensor(name); });
}

std::vector<int64_t> InstancesList::getLenVec() const {
	return vapply<int64_t, InstancesPtr>(*this,
		[=](const InstancesPtr &instance){ return (int64_t)instance->len(); });
}

std::vector<ImageSize> InstancesList::getImageSizes() const {
	return vapply<ImageSize, InstancesPtr>(*this,
		[](const InstancesPtr &instance){ return instance->image_size(); });
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

InstancesList Instances::to(const InstancesList &instance_lists, torch::Device device) {
	InstancesList converted;
	converted.reserve(instance_lists.size());
	for (auto &x : instance_lists) {
		converted.push_back(x->to(device));
	}
	return converted;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Instances::Instances(const ImageSize &image_size, bool check_length) :
	m_image_size(image_size), m_check_length(check_length), m_length(-1) {
}

Instances::Instances(const ImageSize &image_size, SequencePtrMap fields, bool check_length) :
	m_image_size(image_size), m_check_length(check_length), m_length(-1), m_fields(std::move(fields)) {
	for (auto iter : m_fields) {
		setLength(iter.second->size());
	}
}

void Instances::setLength(int len) {
	if (m_check_length) {
		if (m_length < 0) {
			m_length = len;
		}
		else {
			assert(m_length == len);
		}
	}
}

std::string Instances::toString() const {
	string s = "Instances(";
	s += FormatString("num_instances=%d, ", m_length);
	s += FormatString("image_height=%d, ", m_image_size.height);
	s += FormatString("image_width=%d, ", m_image_size.width);
	s += "fields=[";
	for (auto iter : m_fields) {
		s += iter.first + ": ";
		s += iter.second->toString() + ", ";
	}
	s += "])";
	return s;
}

void Instances::set(const std::string &name, const SequencePtr &values) {
	setLength(values->size());
	m_fields[name] = values;
}

void Instances::set(const std::string &name, const torch::Tensor &t) {
	setLength(t.size(0));
	m_fields[name] = make_shared<SequenceTensor>(t);
}

SequencePtr Instances::slice(int64_t start, int64_t end) const {
	assert(m_check_length);
	if (start < 0) start += m_length;
	assert(start >= 0 && start < m_length);

	if (end < 0) end += m_length;
	assert(end >= 0 && end <= m_length);
	
	InstancesPtr ret(new Instances(m_image_size));
	for (auto iter : m_fields) {
		ret->set(iter.first, iter.second->slice(start, end));
	}
	return ret;
}

SequencePtr Instances::index(torch::Tensor item) const {
	assert(m_check_length);
	InstancesPtr ret(new Instances(m_image_size));
	for (auto iter : m_fields) {
		auto tseq = dynamic_pointer_cast<SequenceTensor>(iter.second);
		if (tseq) {
			auto t = tseq->data().index(item);
			ret->set(iter.first, make_shared<SequenceTensor>(t));
		}
	}
	return ret;
}

SequencePtr Instances::cat(const std::vector<SequencePtr> &seqs, int total) const {
	assert(m_check_length);
	int count = seqs.size();
	assert(count > 0);

	auto instances0 = dynamic_pointer_cast<Instances>(seqs[0]);
	if (count == 1) {
		return instances0;
	}

	const auto &image_size = instances0->m_image_size;
	for (auto instances : seqs) {
		auto &size = dynamic_pointer_cast<Instances>(instances)->m_image_size;
		assert(size.height == image_size.height && size.width == image_size.width);
	}
	InstancesPtr ret(new Instances(image_size));
	for (auto iter : instances0->m_fields) {
		auto k = iter.first;
		std::vector<SequencePtr> seqs;
		seqs.reserve(total);
		int total = 0;
		for (int i = 0; i < seqs.size(); i++) {
			auto &seq = dynamic_pointer_cast<Instances>(seqs[i])->m_fields[k];
			total += seq->size();
			seqs.push_back(seq);
		}
		ret->set(k, iter.second->cat(seqs, total));
	}
	return ret;
}

InstancesPtr Instances::to(torch::Device device) const {
	InstancesPtr ret(new Instances(m_image_size));
	for (auto iter : m_fields) {
		auto tseq = dynamic_pointer_cast<SequenceTensor>(iter.second);
		if (tseq) {
			auto t = tseq->data().to(device);
			ret->set(iter.first, make_shared<SequenceTensor>(t));
		}
	}
	return ret;
}
