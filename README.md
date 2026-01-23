# AMLS Assignment 25/26 - BreastMNIST Classification
**Student Number:** SN12345678

## Project Overview
This project benchmarks two machine learning approaches on the BreastMNIST dataset:
1. **Model A:** Support Vector Machine (SVM) comparing raw pixel features vs. HOG (Histogram of Oriented Gradients).
2. **Model B:** A ResNet-18 Deep Neural Network, adapted for single-channel medical images, analyzing the impact of Data Augmentation.

## Requirements
To install dependencies:
`pip install -r requirements.txt`

## File Structure
* `Code/A/model_a.py`: Implementation of SVM with Scikit-Learn pipelines.
* `Code/B/model_b.py`: Implementation of ResNet-18 using PyTorch.
* `Code/data_utils.py`: Shared utilities for loading and transforming MedMNIST data.
* `main.py`: Main script to execute benchmarks and generate results.

## How to Run
Execute the main script from the root directory:
`python main.py`

This will automatically:
1. Create the Dataset/ directory and place the .npz within that directory
2. Train Model A (SVM) with and without HOG features.
3. Train Model B (ResNet) with and without Data Augmentation.
4. Output test accuracy for all experiments.
