# Directories
* Detectron2: converted C++ code
* NetLib2: CUDA .cu compilation

# Environment Variables

* D2_CHECKPOINTS_DIR: where model checkpoint .pkl files (and converted .data files) can be found
* D2_CONFIGS_DEFAULT_DIR: where CfgDefaults.yaml can be found. For example, ".../d2/"
* D2_CONFIGS_DIR: should point to "/configs" under https://github.com/facebookresearch/detectron2 local checkout

Read Detectron2/README.md for how these are set up:
* LIBTORCH_DIR_DEBUG
* LIBTORCH_DIR
* CUDA_PATH
* OPENCV_DIR
* YAML_DIR
