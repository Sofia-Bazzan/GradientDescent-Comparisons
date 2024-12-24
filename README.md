# **BCGD Optimization Methods on Various Datasets**

This project evaluates different optimization methods—**Classic Gradient Descent**, **Block Coordinate Gradient Descent (BCGD) with Randomized Rule**, and **BCGD with Gauss-Southwell Rule**—across three distinct datasets.

## **Overview**

The objective is to compare the performance and convergence behaviors of the specified optimization algorithms on:

1. **Breast Cancer Dataset**: Imported using `sklearn.datasets.load_breast_cancer`, this dataset contains features computed from breast cancer digitized images to classify malignant and benign tumors.

2. **BMI Dataset**: This dataset includes height, weight, and gender information, enabling binary classification to predict gender based on height and weight. The dataset is publicly available on Kaggle.  
   [BMI Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/bmidataset?utm_source=chatgpt.com)

3. **Synthetic Blobs Dataset**: Generated using `make_blobs`, this dataset consists of well-separated clusters with only 5% of the data labeled, simulating semi-supervised learning scenarios.

## **Goals**

- Implement and compare the convergence and accuracy of Classic Gradient Descent and BCGD methods with different update rules.
- Analyze the impact of dataset characteristics on the performance of each optimization method.

## **Methods Implemented**

- **Classic Gradient Descent**: A traditional optimization algorithm that updates parameters in the direction of the negative gradient.
- **BCGD with Randomized Rule**: An optimization method that updates a randomly selected block of coordinates in each iteration.
- **BCGD with Gauss-Southwell Rule**: An optimization technique that selects the block with the maximum gradient magnitude for updating, aiming for more efficient convergence.

## **Datasets**

1. **Breast Cancer Dataset**: Accessible via scikit-learn's `load_breast_cancer` function.
2. **BMI Dataset**: Available on Kaggle, this dataset includes height, weight, and gender information.  
   [BMI Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/bmidataset?utm_source=chatgpt.com)
3. **Synthetic Blobs Dataset**: Created using scikit-learn's `make_blobs` function, with 5% labeled data to mimic semi-supervised learning conditions.

## **Key Insights**

- Evaluated the convergence rates and stability of each optimization method across different datasets.
- Assessed the influence of dataset properties, such as feature distribution and labeling, on algorithm performance.
- Provided a comparative analysis based on metrics like accuracy, convergence time, and computational efficiency.

## **Authors**

- **Your Name** (Your Email)
- **Your Colleague's Name** (Colleague's Email)

## **Contact**

For questions or feedback, please open an issue in this repository.
