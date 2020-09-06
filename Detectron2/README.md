# License and Disclaimers

* Apache License Version 2.0
* All occurrences of "Detectron2" in this code base is ONLY for indicating its origin was from Facebook's Detectron2
sourcec code. "Detectron2" is a trademark of Facebook.
* No liability.
* No tech support or correspondences of any inquiries. This is merely a spare-time fun project for me.
* Contribution is more than welcomed though.
* Only did it on Windows with Visual Studio 2017. More OS and dev env support are welcomed.
* Inferences were tested with 5 demos, but not training. Therefore, there may be quite a few bugs during conversion
in training code paths. Feel free to submit bugs or fixes along these.

# Building "Detectron2 C++"

## Dependencies

* [libtorch 1.5.1](https://pytorch.org/)
```
1. PyTorch Build: 1.5.1  You might have to manually modify generated URL to choose this version number.
2. OS: Windows
3. Package: LibTorch
4. Language: C++
5. CUDA: 10.2

Download, expand, then create environment variables to point to root directories:
1. LIBTORCH_DIR_DEBUG for debug version
2. LIBTORCH_DIR for release version.
```

* [cuda 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)
```
CUDA will create CUDA_PATH automatically for you.
```

* [opencv](https://github.com/opencv/opencv)
```
Create OPENCV_DIR to point to "{opencv_root}\build_opencv\install\x64\vc15"
```

* [yaml](https://github.com/jbeder/yaml-cpp)
```
1. git clone https://github.com/jbeder/yaml-cpp.git
2. Use CMake (from kitware.com) to configure: use **{yaml_root}\build** as build directory.
3. Still within CMake, click on **Generate** to generate Visual Studio solution file.
4. Open the generated Visual Studio solution file. Build libs from here.
5. Define **YAML_DIR** environment variable to be yaml's root directory {yaml_root}.
```

## Running demo

* Import model checkpoints: For example,
```
Download those .pkl files from https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md and
put them into $(D2_CHECKPOINTS_DIR). Then, run commands:

> cd Detectron2\Import
> python3 ImportBaseline.py model_final_997cc7

This will create model_final_997cc7.data file under $(D2_CHECKPOINTS_DIR) and you need it to load a model.
```

* Write code like this,
```
#include <Detectron2/Detectron2Includes.h>

void demo() {
	int selected = 4; // <-- change this number to choose different demo

	static const char *models[] = {
		"COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
		"COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl",
		"COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl",
		"COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl",
		"COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl"
	};
	string model = models[selected];
	auto tokens = tokenize(model, '/');

	VisualizationDemo::Options options;
	assert(configDir);
	options.config_file = File::ComposeFilename(configDir, tokens[0] + "\\" + tokens[1] + ".yaml");
	options.webcam = true;
	options.opts = { {"MODEL.WEIGHTS", YAML::Node("detectron2://" + model) } };
	try {
		VisualizationDemo::start(options);
	}
	catch (const std::exception &e) {
		const char *msg = e.what();
		std::cerr << msg;
	}
}
```

# Python => C++ Conversion

## version

* [tag: v0.1.3](https://github.com/facebookresearch/detectron2)

## skipping

* checkpoint/
* config/
* data/
* engine/
* evaluation/
* export/
* model_zoo/
* solver/
* utils/

* modeling/test_time_augmentation.py
* modeling/meta_arch/retinanet.py

## fully converted

* fvcore/common/config.py
* fvcore/nn/smooth_l1_loss.py
* data/datasets/builtin.py
* data/datasets/builtin_meta.py
* data/datasets/register_coco.py
* demo/demo.py
* demo/predictor.py
* layers/
* modeling/
* structures/
* utils/video_visualizer.py
* utils/visualizer.py

## partially converted

* fvcore/transforms/transform.py
* data/transform/transform.py
* data/transform/transform_gen.py
* data/catalog.py
* utisl/events.py
* engine/defaults.py

# Installation of "Detectron2 Python" on Windows

## Install detectron2
```
git clone https://github.com/facebookresearch/detectron2.git
pip3 install -e detectron2
```

## Install opencv
```
pip3 install opencv-python
```

## Install pycocotools

Instruction from [this web page](https://github.com/matterport/Mask_RCNN/issues/6)

```
pip3 install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
```

## Running demo
```
cd detetron2\demo
python3 demo.py --config-file "../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" --webcam \
	--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```
