#pragma once

#include "Sequence.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from structures/instances.py

	class Instances;
	using InstancesPtr = std::shared_ptr<Instances>;
	class InstancesList : public std::vector<InstancesPtr> {
	public:
		TensorVec getTensorVec(const std::string &name) const;
		std::vector<int64_t> getLenVec() const;
		std::vector<ImageSize> getImageSizes() const;
	};

	using InstancesMap = std::unordered_map<std::string, InstancesPtr>;
	using InstancesMapList = std::vector<std::shared_ptr<InstancesMap>>;

	/**
		This class represents a list of instances in an image.
		It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
		All fields must have the same ``__len__`` which is the number of instances.

		All other (non-field) attributes of this class are considered private:
		they must start with '_' and are not modifiable by a user.

		Some basic usage:

		1. Set/Get a field:

		   .. code-block:: python

			  instances.gt_boxes = Boxes(...)
			  print(instances.pred_masks)  # a tensor of shape (N, H, W)
			  print('gt_masks' in instances)

		2. ``len(instances)`` returns the number of instances
		3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
		   and returns a new :class:`Instances`.
		   Typically, ``indices`` is a integer vector of indices,
		   or a binary mask of length ``num_instances``,
	*/
	class Instances : public Sequence {
	public:
		static InstancesList to(const InstancesList &instance_lists, torch::Device device);
		template<typename T>
		static InstancesList to(const std::vector<T> &items, torch::Device device,
			std::function<InstancesPtr(const T*)> fp) {
			InstancesList ret;
			ret.reserve(items.size());
			for (auto &item : items) {
				ret.push_back(fp(&item)->to(device));
			}
			return ret;
		}

	public:
		// image_size (height, width): the spatial size of the image.
		// check_length: true: parallel vectors; false: heterogeneous map
		Instances(const ImageSize &image_size, bool check_length = true);
		Instances(const ImageSize &image_size, SequencePtrMap fields, bool check_length = true);

		ImageSize image_size() const { return m_image_size; }

		// Set the field named `name` to `value`. The length of `value` must be the number of instances,
		// and must agree with other existing fields in this object.
		void set(const std::string &name, const SequencePtr &values);
		void set(const std::string &name, const torch::Tensor &t);

		//  bool: whether the field called `name` exists.
		bool has(const std::string &name) const {
			return m_fields.find(name) != m_fields.end();
		}

		// Remove the field called `name`.
		void remove(const std::string &name) {
			m_fields.erase(name);
		}

		// Returns the field called `name`.
		const SequencePtr &get(const std::string &name) const {
			auto iter = m_fields.find(name);
			assert(iter != m_fields.end());
			return iter->second;
		}
		const SequencePtr &operator[](const std::string &name) const {
			return get(name);
		}
		template<typename T>
		const std::vector<T> &getVec(const std::string &name) const {
			return std::dynamic_pointer_cast<SequenceVec<T>>(get(name))->data();
		}
		torch::Tensor getTensor(const std::string &name) const {
			return std::dynamic_pointer_cast<SequenceTensor>(get(name))->data();
		}

		// Returns: dict: a dict which maps names (str) to data of the fields
		// Modifying the returned dict will modify this instance.
		const SequencePtrMap &get_fields() {
			return m_fields;
		}
			
		// Modifying the returned dict will modify this instance.
		SequencePtrMap move_fields() {
			return std::move(m_fields);
		}

		// Returns: Instances: all fields are called with a `to(device)`, if the field has this method.
		InstancesPtr to(torch::Device device) const;

		/*
			Args:
				item: an index-like object and will be used to index all the fields.

			Returns:
				If `item` is a string, return the data in the corresponding field.
				Otherwise, returns an `Instances` where all fields are indexed by `item`.
		*/
		InstancesPtr operator[](int item) const { return std::dynamic_pointer_cast<Instances>(index(item)); }
		InstancesPtr operator[](torch::Tensor item) const { return std::dynamic_pointer_cast<Instances>(index(item)); }

		int len() { assert(m_check_length); return m_length; }
		virtual int size() const { assert(m_check_length); return m_length; }
		virtual std::string toString() const;
		virtual SequencePtr slice(int64_t start, int64_t end) const;
		virtual SequencePtr index(int item) const { assert(m_check_length); return slice(item, item + 1); }
		virtual SequencePtr index(torch::Tensor item) const;
		virtual SequencePtr cat(const std::vector<SequencePtr> &seqs, int total) const;

	private:
		ImageSize m_image_size;
		SequencePtrMap m_fields;

		bool m_check_length;
		int m_length;
		void setLength(int len);
	};
}
