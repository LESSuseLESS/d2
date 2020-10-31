#pragma once

#include <Detectron2/Structures/Instances.h>
#include <Detectron2/Utils/Predictor.h>
#include <Detectron2/Utils/VideoAnalyzer.h>
#include <Detectron2/Utils/Visualizer.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from demo/demo.py
	// converted from demo/predictor.py
	
	class VisualizationDemo {
	public:
		struct Options {
			std::string config_file					// path to config file
				= "configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml";
			bool webcam = false;					// Take inputs from webcam
			std::string video_input;				// Path to video file
			std::vector<std::string> input;			// A list of space separated input images
													// or a single glob pattern such as 'directory/*.jpg'
			std::string output;						// A file or directory to save output visualizations.
													// If not given, will show output in an OpenCV window.
			CfgNode::OptionList opts;				// Modify config options using the command-line 'KEY VALUE' pairs
			float confidence_threshold = 0.5; 		// Minimum score for instance predictions to be shown
		};
		static void start(const Options &options);

		static CfgNode setup_cfg(const std::string &config_file, const CfgNode::OptionList &opts,
			float confidence_threshold);
	public:
		/**
			cfg (CfgNode):
			instance_mode (ColorMode):
			parallel (bool): whether to run the model in different processes from visualization.
				Useful since the visualization logic can be slow.
		*/
		VisualizationDemo(const CfgNode &cfg, ColorMode instance_mode = ColorMode::kIMAGE, bool parallel = false);

		/**
			Args:
				image (np.ndarray): an image of shape (H, W, C) (in BGR order).
					This is the format used by OpenCV.

			Returns:
				predictions (dict): the output of the model.
				vis_output (VisImage): the visualized image output.
		*/
		std::tuple<InstancesPtr, VisImage> run_on_image(torch::Tensor image);

		/**
			Visualizes predictions on frames of the input video.

			Args:
				video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
					either a webcam or a video file.

			Yields:
				ndarray: BGR visualizations of each video frame.
		*/
		void run_on_video(cv::VideoCapture &video, std::function<bool(cv::Mat)> vis_frame_processor);

		/**
			Captures predictions from frames of the input video.
		 */
		void analyze_on_video(cv::VideoCapture &video, VideoAnalyzer &analyzer);

	private:
		Metadata m_metadata;
		torch::Device m_cpu_device;
		ColorMode m_instance_mode;
		bool m_parallel;
		std::shared_ptr<Predictor> m_predictor;
	};
}
