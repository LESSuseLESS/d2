#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	class EventStorage;

    /**
		Returns:
			The :class:`EventStorage` object that's currently being used.
			Throws an error if no :class:`EventStorage` is currently enabled.
	*/
	EventStorage &get_event_storage();

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Base class for writers that obtain events from :class:`EventStorage` and process them.
	class EventWriter {
	public:
		virtual ~EventWriter() {}

		virtual void write() = 0;
		virtual void close() = 0;
	};

    /**
		Write scalars to a json file.

		It saves scalars as one json per line (instead of a big json) for easy parsing.

		Examples parsing such a json file:

		.. code-block:: none

			$ cat metrics.json | jq -s '.[0:2]'
			[
			  {
				"data_time": 0.008433341979980469,
				"iteration": 20,
				"loss": 1.9228371381759644,
				"loss_box_reg": 0.050025828182697296,
				"loss_classifier": 0.5316952466964722,
				"loss_mask": 0.7236229181289673,
				"loss_rpn_box": 0.0856662318110466,
				"loss_rpn_cls": 0.48198649287223816,
				"lr": 0.007173333333333333,
				"time": 0.25401854515075684
			  },
			  {
				"data_time": 0.007216215133666992,
				"iteration": 40,
				"loss": 1.282649278640747,
				"loss_box_reg": 0.06222952902317047,
				"loss_classifier": 0.30682939291000366,
				"loss_mask": 0.6970193982124329,
				"loss_rpn_box": 0.038663312792778015,
				"loss_rpn_cls": 0.1471673548221588,
				"lr": 0.007706666666666667,
				"time": 0.2490077018737793
			  }
			]

			$ cat metrics.json | jq '.loss_mask'
			0.7126231789588928
			0.689423680305481
			0.6776131987571716
			...

	*/
	class JSONWriter : public EventWriter {
	public:
		/**
			Args:
				json_file (str): path to the json file. New data will be appended if the file exists.
				window_size (int): the window size of median smoothing for the scalars whose
					`smoothing_hint` are True.
		*/
		JSONWriter(const std::string &json_file, int window_size = 20);

		virtual void write() override {}
		virtual void close() override {}
	};

	//  Write all scalars to a tensorboard file.
	class TensorboardXWriter : public EventWriter {
	public:
		/**
			Args:
				log_dir (str): the directory to save the output events
				window_size (int): the scalars will be median-smoothed by this window size

				kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
		*/
		TensorboardXWriter(const std::string &log_dir, int window_size);

		virtual void write() override {}
		virtual void close() override {}
	};

    /**
		Print **common** metrics to the terminal, including
		iteration time, ETA, memory, all losses, and the learning rate.

		To print something different, please implement a similar printer by yourself.
	*/
	class CommonMetricPrinter : public EventWriter {
	public:
		/**
			Args:
				max_iter (int): the maximum number of iterations to train.
					Used to compute ETA.
		*/
		CommonMetricPrinter(int max_iter);

		virtual void write() override {}
		virtual void close() override {}
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
		The user-facing class that provides metric storage functionalities.

		In the future we may add support for storing / logging other types of data if needed.
	*/
	class EventStorage {
	public:
		/**
			Args:
				start_iter (int): the iteration number to start with
		*/
		EventStorage(int start_iter = 0) : m_iter(start_iter) {}

		/**
			Add an `img_tensor` associated with `img_name`, to be shown on
			tensorboard.

			Args:
				img_name (str): The name of the image to put into tensorboard.
				img_tensor (torch.Tensor or numpy.array): An `uint8` or `float`
					Tensor of shape `[channel, height, width]` where `channel` is
					3. The image format should be RGB. The elements in img_tensor
					can either have values in [0, 1] (float32) or [0, 255] (uint8).
					The `img_tensor` will be visualized in tensorboard.
		*/
		void put_image(const std::string &img_name, const torch::Tensor &img_tensor) {}

		/**
			Add a scalar `value` to the `HistoryBuffer` associated with `name`.

			Args:
				smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
					smoothed when logged. The hint will be accessible through
					:meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
					and apply custom smoothing rule.

					It defaults to True because most scalars we save need to be smoothed to
					provide any useful signal.
		*/
		void put_scalar(const std::string &img_name, torch::Scalar value, bool smoothing_hint = true) {}

		/**
			Put multiple scalars from keyword arguments.

			Examples:

				storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
		*/
		void put_scalars(const std::unordered_map<std::string, torch::Scalar> &values, bool smoothing_hint = true) {}

		/**
			Create a histogram from a tensor.

			Args:
				hist_name (str): The name of the histogram to put into tensorboard.
				hist_tensor (torch.Tensor): A Tensor of arbitrary shape to be converted
					into a histogram.
				bins (int): Number of histogram bins.
		*/
		void put_histogram(const std::string &hist_name, const torch::Tensor &hist_tensor, int bins = 1000) {}

		/**
			Returns:
				HistoryBuffer: the scalar history for name
		*/
		void history(const std::string &name) {}

		/**
			Returns:
				dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
		*/
		void histories() {}

		/**
			Returns:
				dict[name -> number]: the scalars that's added in the current iteration.
		*/
		void latest() {}

		/**
			Similar to :meth:`latest`, but the returned values
			are either the un-smoothed original latest value,
			or a median of the given window_size,
			depend on whether the smoothing_hint is True.

			This provides a default behavior that other writers can use.
		*/
		void latest_with_smoothing_hint(int window_size = 20) {}

		/**
			Returns:
				dict[name -> bool]: the user-provided hint on whether the scalar
					is noisy and needs smoothing.
		*/
		void smoothing_hints() {}

		/**
			User should call this function at the beginning of each iteration, to
			notify the storage of the start of a new iteration.
			The storage will then be able to associate the new data with the
			correct iteration number.
		*/
		void step() {}
		int iter() const { return m_iter; }

		/**
			Yields:
				A context within which all the events added to this storage
				will be prefixed by the name scope.
		*/
		void push_name_scope(const std::string &name) {}
		void pop_name_scope() {}

		/**
			Delete all the stored images for visualization. This should be called
			after images are written to tensorboard.
		*/
		void clear_images() {}

		/**
			Delete all the stored histograms for visualization.
			This should be called after histograms are written to tensorboard.
		*/
		void clear_histograms() {}

	private:
		int m_iter;
	};
}
