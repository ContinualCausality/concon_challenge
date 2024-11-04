# ConCon: Continual Confounded Dataset

## Overview
This dataset is designed for **continual learning** in confounded environments, challenging models to overcome task-specific confounders and generalize to unconfounded data.
The ConCon dataset is a continually confounded dataset built on top of CLEVR, an image rendering dataset using Blender. This dataset is designed for training machine learning models that can handle confounding factors in a continually changing environment. 
In this repository we provide a starter template for the ConCon challenge. 

## Getting Started
You can download the dataset at [link](give link here).

We provide Docker image to setup the development enviroment. To run the docker:

    cd .docker/
    docker build -t name_for_docker_cont .
    docker run -it --rm  --gpus device=00 -v /path_to_project:/workspace name_for_docker_cont

We also provide a starter template to train a Resnet-18 model on the tasks of ConCon dataset in a naive sequential manner. 

    python train.py --epochs $EPOCHS --batch_size $BATCHSIZE --model_name $MODEL --dataset_type $DATASET_TYPE \
        --train_path_task0 $TRAIN_PATH_TASK0 --train_path_task1 $TRAIN_PATH_TASK1 --train_path_task2 $TRAIN_PATH_TASK2 \
        --val_path_task0 $VAL_PATH_TASK0 --val_path_task1 $VAL_PATH_TASK1 --val_path_task2 $VAL_PATH_TASK2 \
        --test_path_task0 $TEST_PATH_TASK0 --test_path_task1 $TEST_PATH_TASK1 --test_path_task2 $TEST_PATH_TASK2 \
        --test_path_global $TEST_PATH_GLOBAL --seed $SEED --results_dir $ROOT_DIR
 
Here:

``dataset_type`` would correspond to either ``strict`` or ``disjoint``

``train_path_task0`` specifies the path for task 1 dataset 

``val_path_task0`` specifies the path for task 1 dataset 

``test_path_task0`` specifies the path for task 3 dataset 

``test_path_global`` specifies path to the unconfounded dataset


## Dataset Structure

The dataset is organized into 2 variants containing three tasks each, with each task introducing different confounding factors that models need to overcome. 
The primary objective is to train models capable of generalizing across these tasks by ignoring the task-specific confounders and learning the underlying ground truth.

The dataset includes:

### case_disjoint_main
- **Purpose**: This folder contains the **disjoint confounder** scenario, where the confounders across tasks are **mutually exclusive**. Each task is confounded by a unique feature, ensuring no overlap in confounders between tasks.
- **Structure**: The folder is similarly divided into `train/`, `val/`, and `test/` subfolders, corresponding to training, validation, and testing sets for this disjoint setup.
Each subfolders contain 3 tasks `t0/`, `t1/`, and `t2/` with corresponding folders `0/` and `1/` for negative and positive labels.
- **Usage**: Ideal for testing models in situations where confounders differ drastically between tasks, making generalization across tasks more challenging.

### case_strict_main
- **Purpose**: This folder contains the **strict confounder** scenario, where the confounders are more persistent across tasks. Unlike the disjoint setup, the confounders in this case may overlap between tasks, making it harder for the model to distinguish the true signal from the spurious associations.
- **Structure**: The folder is similarly divided into `train/`, `val/`, and `test/` subfolders, corresponding to training, validation, and testing sets for this disjoint setup.
Each subfolders contain 3 tasks `t0/`, `t1/`, and `t2/` with corresponding folders `0/` and `1/` for negative and positive labels.
- **Usage**: This case tests the model's ability to handle more challenging confounders that recur across different tasks.

### unconfounded
- **Purpose**: This folder contains **unconfounded data** that should be used strictly for final evaluation.
- **Structure**: It contains a `test/` subfolder for evaluation. **Do not use the `train/` and `val/` sets in this folder for training or validation**—they are provided for informational purposes only.
- **Usage**: After training your model on the confounded tasks, you must test it on the data in `unconfounded/test/` to assess how well it generalizes when no confounders are present. Using any part of the `train/` or `val/` sets for training or tuning will invalidate your results.


## Programming Language and Frameworks
Participants are free to use **any programming language or framework** (e.g., Python, PyTorch, TensorFlow) for developing their models. 






