#pragma once

#include <Detectron2/Data/MetadataCatalog.h>
#include <Detectron2/Structures/Instances.h>
#include "VisImage.h"

namespace Detectron2
{
	class BitMasks;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from utils/visualizer.py

	class Visualizer {
	public:
		/**
			Args:
				classes (list[int] or None):
				scores (list[float] or None):
				class_names (list[str] or None):

			Returns:
				list[str] or None
		*/
		static std::vector<std::string> _create_text_labels(const torch::Tensor &classes, const torch::Tensor &scores,
			const std::vector<ClassColor> &class_colors = {});

	public:
		/**
			img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
				the height and width of the image respectively. C is the number of
				color channels. The image is required to be in RGB format since that
				is a requirement of the Matplotlib library. The image is also expected
				to be in the range [0, 255].
			metadata (MetadataCatalog): image metadata.
		*/
		Visualizer(const torch::Tensor &img_rgb, Metadata metadata, float scale = 1.0,
			ColorMode instance_mode = ColorMode::kIMAGE);

		/**
			Returns:
				output (VisImage): the image output containing the visualizations added
				to the image.
		*/
		VisImage get_output() { return m_output; }
		int height() { return m_output.height(); }
		int width() { return m_output.width(); }

		void set_grayscale_image(const torch::Tensor &img);

		/**
			Draw instance-level prediction results on an image.

			Args:
				predictions (Instances): the output of an instance detection/segmentation
					model. Following fields will be used to draw:
					"pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

			Returns:
				output (VisImage): image object with visualizations.
		*/
		VisImage draw_instance_predictions(const InstancesPtr &predictions);

		/**
			Draw semantic segmentation predictions/labels.

			Args:
				sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
					Each value is the integer label of the pixel.
				area_threshold (int): segments with less than `area_threshold` are not drawn.
				alpha (float): the larger it is, the more opaque the segmentations are.

			Returns:
				output (VisImage): image object with visualizations.
		*/
		VisImage draw_sem_seg(const torch::Tensor &sem_seg, int area_threshold = 0, float alpha = 0.8);

		/**
			Draw panoptic prediction results on an image.

			Args:
				panoptic_seg (Tensor): of shape (height, width) where the values are ids for each
					segment.
				segments_info (list[dict]): Describe each segment in `panoptic_seg`.
					Each dict contains keys "id", "category_id", "isthing".
				area_threshold (int): stuff segments with less than `area_threshold` are not drawn.

			Returns:
				output (VisImage): image object with visualizations.
		*/
		VisImage draw_panoptic_seg_predictions(const torch::Tensor &panoptic_seg,
			const std::vector<SegmentInfo> &segments_info, int area_threshold = 0, float alpha = 0.7);

		/**
			Draw annotations/segmentaions in Detectron2 Dataset format.

			Args:
				dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

			Returns:
				output (VisImage): image object with visualizations.
		*/
		VisImage draw_dataset_dict(int dic);

		/**
			Args:
				boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
					or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
					or a :class:`RotatedBoxes`,
					or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
					for the N objects in a single image,
				labels (list[str]): the text to be displayed for each instance.
				masks (masks-like object): Supported types are:

					* :class:`detectron2.structures.PolygonMasks`,
					  :class:`detectron2.structures.BitMasks`.
					* list[list[ndarray]]: contains the segmentation masks for all objects in one image.
					  The first level of the list corresponds to individual instances. The second
					  level to all the polygon that compose the instance, and the third level
					  to the polygon coordinates. The third level should have the format of
					  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
					* list[ndarray]: each ndarray is a binary mask of shape (H, W).
					* list[dict]: each dict is a COCO-style RLE.
				keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
					where the N is the number of instances and K is the number of keypoints.
					The last dimension corresponds to (x, y, visibility or score).
				assigned_colors (list[matplotlib.colors]): a list of colors, where each color
					corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
					for full list of formats that the colors are accepted in.

			Returns:
				output (VisImage): image object with visualizations.
		*/
		VisImage overlay_instances(
			const torch::Tensor &boxes,
			const std::vector<std::string> &labels = {},
			const std::vector<std::shared_ptr<GenericMask>> &masks = {},
			const torch::Tensor &keypoints = torch::Tensor(),
			const std::vector<VisColor> &assigned_colors = {},
			float alpha = 0.5);

		/**
			Args:
				boxes (ndarray): an Nx5 numpy array of
					(x_center, y_center, width, height, angle_degrees) format
					for the N objects in a single image.
				labels (list[str]): the text to be displayed for each instance.
				assigned_colors (list[matplotlib.colors]): a list of colors, where each color
					corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
					for full list of formats that the colors are accepted in.

			Returns:
				output (VisImage): image object with visualizations.
		*/
		VisImage overlay_rotated_instances(const torch::Tensor &boxes = torch::Tensor(),
			const std::vector<std::string> &labels = {}, const std::vector<VisColor> &assigned_colors = {});

		/**
			Draws keypoints of an instance and follows the rules for keypoint connections
			to draw lines between appropriate keypoints. This follows color heuristics for
			line color.

			Args:
				keypoints (Tensor): a tensor of shape (K, 3), where K is the number of keypoints
					and the last dimension corresponds to (x, y, probability).

			Returns:
				output (VisImage): image object with visualizations.
		*/
		VisImage draw_and_connect_keypoints(const torch::Tensor &keypoints);

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Primitive drawing functions

		/**
			Args:
				text (str): class label
				position (tuple): a tuple of the x and y coordinates to place text on image.
				font_size (int, optional): font of the text. If not provided, a font size
					proportional to the image width is calculated and used.
				color: color of the text. Refer to `matplotlib.colors` for full list
					of formats that are accepted.
				horizontal_alignment (str): see `matplotlib.text.Text`
				rotation: rotation angle in degrees CCW

			Returns:
				output (VisImage): image object with text drawn.
		*/
		VisImage draw_text(const std::string &text, const Pos &position, float font_size = 0,
			const VisColor &color = {}, const Canvas::Alignment &horizontal_alignment = Canvas::kCenter,
			float rotation = 0);

		/**
			Args:
				box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
					are the coordinates of the image's top left corner. x1 and y1 are the
					coordinates of the image's bottom right corner.
				alpha (float): blending efficient. Smaller values lead to more transparent masks.
				edge_color: color of the outline of the box. Refer to `matplotlib.colors`
					for full list of formats that are accepted.
				line_style (string): the string to use to create the outline of the boxes.

			Returns:
				output (VisImage): image object with box drawn.
		*/
		VisImage draw_box(const torch::Tensor &box_coord, float alpha = 0.5,
			const VisColor &edge_color = {}, const Canvas::LineStyle &line_style = Canvas::kSolid);

		/**
			Args:
				rotated_box (tuple): a tuple containing (cnt_x, cnt_y, w, h, angle),
					where cnt_x and cnt_y are the center coordinates of the box.
					w and h are the width and height of the box. angle represents how
					many degrees the box is rotated CCW with regard to the 0-degree box.
				alpha (float): blending efficient. Smaller values lead to more transparent masks.
				edge_color: color of the outline of the box. Refer to `matplotlib.colors`
					for full list of formats that are accepted.
				line_style (string): the string to use to create the outline of the boxes.
				label (string): label for rotated box. It will not be rendered when set to None.

			Returns:
				output (VisImage): image object with box drawn.
		*/
		VisImage draw_rotated_box_with_label(const torch::Tensor &rotated_box, float alpha = 0.5,
			const VisColor &edge_color = {}, const Canvas::LineStyle &line_style = Canvas::kSolid,
			const std::string &label = "");

		/**
			Args:
				circle_coord (list(int) or tuple(int)): contains the x and y coordinates
					of the center of the circle.
				color: color of the polygon. Refer to `matplotlib.colors` for a full list of
					formats that are accepted.
				radius (int): radius of the circle.

			Returns:
				output (VisImage): image object with box drawn.
		*/
		VisImage draw_circle(const Pos &circle_coord, const VisColor &color, int radius = 3);

		/**
			Args:
				x_data (list[int]): a list containing x values of all the points being drawn.
					Length of list should match the length of y_data.
				y_data (list[int]): a list containing y values of all the points being drawn.
					Length of list should match the length of x_data.
				color: color of the line. Refer to `matplotlib.colors` for a full list of
					formats that are accepted.
				linestyle: style of the line. Refer to `matplotlib.lines.Line2D`
					for a full list of formats that are accepted.
				linewidth (float or None): width of the line. When it's None,
					a default value will be computed and used.

			Returns:
				output (VisImage): image object with line drawn.
		*/
		VisImage draw_line(const std::vector<int> &x_data, const std::vector<int> &y_data,
			const VisColor &color, const Canvas::LineStyle &linestyle = Canvas::kSolid, float linewidth = 0);

		/**
			Args:
				binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
					W is the image width. Each value in the array is either a 0 or 1 value of uint8
					type.
				color: color of the mask. Refer to `matplotlib.colors` for a full list of
					formats that are accepted. If None, will pick a random color.
				edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
					full list of formats that are accepted.
				text (str): if None, will be drawn in the object's center of mass.
				alpha (float): blending efficient. Smaller values lead to more transparent masks.
				area_threshold (float): a connected component small than this will not be shown.

			Returns:
				output (VisImage): image object with mask drawn.
		*/
		VisImage draw_binary_mask(torch::Tensor binary_mask, VisColor color = {},
			const VisColor &edge_color = {}, const std::string &text = "", float alpha = 0.5,
			float area_threshold = 4096);

		/**
			Args:
				segment: numpy array of shape Nx2, containing all the points in the polygon.
				color: color of the polygon. Refer to `matplotlib.colors` for a full list of
					formats that are accepted.
				edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
					full list of formats that are accepted. If not provided, a darker shade
					of the polygon color will be used instead.
				alpha (float): blending efficient. Smaller values lead to more transparent masks.

			Returns:
				output (VisImage): image object with polygon drawn.
		*/
		VisImage draw_polygon(const torch::Tensor &segment, VisColor color, VisColor edge_color = {},
			float alpha = 0.5);

	private:
		torch::Tensor m_img;
		Metadata m_metadata;
		VisImage m_output;
		torch::Device m_cpu_device;
		float m_default_font_size;
		ColorMode m_instance_mode;

		/**
			Create a grayscale version of the original image.
			The colors in masked area, if given, will be kept.
		*/
		torch::Tensor _create_grayscale_image(const torch::Tensor &mask = torch::Tensor());
	};
}
