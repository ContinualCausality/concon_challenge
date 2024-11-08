# ConCon: Continual Confounded Dataset

## Overview
This dataset is designed for **continual learning** in confounded environments, challenging models to overcome task-specific confounders and generalize to unconfounded data.
The ConCon dataset is a continually confounded dataset built on top of CLEVR, an image rendering dataset using Blender. This dataset is designed for training machine learning models that can handle confounding factors in a continually changing environment. 
In this repository we provide a starter template for the ConCon challenge. 

## Getting Started
You can download the dataset at [here](https://www.kaggle.com/competitions/concon).

We provide Docker image to setup the development enviroment. To run the docker:

    cd .docker/
    docker build -t name_for_docker_cont -f Dockerfile .
    docker run -it --rm  --gpus device=00 -v /path_to_project:/workspace name_for_docker_cont

We also provide a starter template to train a Resnet-18 model on the tasks of ConCon dataset in a naive sequential manner. 

    python src/train.py --epochs $EPOCHS --batch_size $BATCHSIZE --model_name $MODEL --dataset_type $DATASET_TYPE --dataset_path $DATASET_PATH 
    --test_path_global $TEST_PATH_GLOBAL --seed $SEED --results_dir $ROOT_DIR
 

Once you have configured your kaggle key, you can download the dataset by running ``kaggle competitions download -c concon``.
For example, if your dataset lies in the same folder as the train.py file, you can then run:

    python src/train.py --epochs 50 --batch_size 64 --model_name Resnet --dataset_type strict --dataset_path /strict 
    --test_path_global /unconf/test/t0 --seed 42 --results_dir /project

to train and evaluate on the strict dataset.

You can then submit the generated csv file containing the predicted labels to kaggle by running ``kaggle competitions submit -c concon -f labels.csv -m "Message"``.

## Dataset Structure

The dataset is organized into 2 variants containing three tasks each, with each task introducing different confounding factors that models need to overcome. 
The primary objective is to train models capable of generalizing across these tasks by ignoring the task-specific confounders and learning the underlying ground truth.

The dataset includes:

### disjoint
- **Purpose**: This folder contains the **disjoint confounder** scenario, where the confounders across tasks are **mutually exclusive**. Each task is confounded by a unique feature, ensuring no overlap in confounders between tasks.
- **Structure**: The folder is similarly divided into `train/`, `val/`, and `test/` subfolders, corresponding to training, validation, and testing sets for this disjoint setup.
Each subfolders contain 3 tasks `t0/`, `t1/`, and `t2/` with corresponding folders `0/` and `1/` for negative and positive labels.
- **Usage**: Ideal for testing models in situations where confounders differ drastically between tasks, making generalization across tasks more challenging.

### strict
- **Purpose**: This folder contains the **strict confounder** scenario, where the confounders are more persistent across tasks. Unlike the disjoint setup, the confounders in this case may overlap between tasks, making it harder for the model to distinguish the true signal from the spurious associations.
- **Structure**: The folder is similarly divided into `train/`, `val/`, and `test/` subfolders, corresponding to training, validation, and testing sets for this disjoint setup.
Each subfolders contain 3 tasks `t0/`, `t1/`, and `t2/` with corresponding folders `0/` and `1/` for negative and positive labels.
- **Usage**: This case tests the model's ability to handle more challenging confounders that recur across different tasks.

### unconf
- **Purpose**: This folder contains **unconfounded data** that should be used strictly for final evaluation.
- **Structure**: It contains a `test/` subfolder for evaluation.
- **Usage**: After training your model on the confounded tasks, you must test it on the data in `unconf/test/t0` to assess how well it generalizes when no confounders are present. 

## Programming Language and Frameworks
Participants are free to use **any programming language or framework** (e.g., Python, PyTorch, TensorFlow) for developing their models. 






