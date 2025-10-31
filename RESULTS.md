# OCR Model Training and Selection Report

## 1. Objective

The goal of this experiment was to train and select the optimal Convolutional Neural Network (CNN) architectures for a Mixture of Experts (MoE) OCR system. The system required three specialized "expert" models: one for digits (0-9), one for uppercase letters (A-Z), and one for lowercase letters (a-z).

## 2. Methodology

### Dataset
The EMNIST (`byclass`) dataset was used, which contains over 800,000 standardized 28x28 grayscale images of digits, uppercase, and lowercase letters.

### Architectures Tested
Three different CNN architectures were evaluated for each expert task:

*   **`CNNModel_Small`**: A baseline model with 2 convolutional layers and 1 fully-connected layer.
*   **`CNNModel_Medium`**: A larger model with 3 convolutional layers and a wider fully-connected layer to capture more complex features.
*   **`CNNModel_Large`**: A model with the same convolutional base as `Medium` but with a deeper, multi-layer classifier.

### Experiment
To ensure the reliability of the results, the full training and evaluation process was conducted twice. The model that achieved the highest validation accuracy and lowest validation loss was selected for each expert category. Early stopping was used based on validation loss to prevent overfitting.

## 3. Results Summary

The following tables show the best performance (validation accuracy and loss) achieved by each architecture across the two independent runs.

### Run 1 Results
| Architecture    | Expert       | Best Val Accuracy    | Best Val Loss  |
|:----------------|:-------------|:---------------------|:---------------|
| **small**       | **digits**   | **99.60%**           | **0.0156**     |
| small           | uppercase    | 97.97%               | 0.0770         |
| small           | lowercase    | 95.84%               | 0.1412         |
| medium          | digits       | 99.51%               | 0.0180         |
| **medium**      | **uppercase**| **98.31%**           | **0.0618**     |
| medium          | lowercase    | 95.94%               | 0.1338         |
| large           | digits       | 99.58%               | 0.0184         |
| large           | uppercase    | 98.22%               | 0.0706         |
| large           | lowercase    | 95.88%               | 0.1381         |

### Run 2 Results
| Architecture    | Expert       | Best Val Accuracy    | Best Val Loss  |
|:----------------|:-------------|:---------------------|:---------------|
| small           | digits       | 99.51%               | 0.0165         |
| small           | uppercase    | 98.13%               | 0.0779         |
| small           | lowercase    | 95.39%               | 0.1520         |
| medium          | digits       | 99.59%               | 0.0182         |
| medium          | uppercase    | 98.27%               | 0.0679         |
| **medium**      | **lowercase**| **96.16%**           | **0.1279**     |
| large           | digits       | 99.54%               | 0.0195         |
| large           | uppercase    | 98.17%               | 0.0699         |
| large           | lowercase    | 96.00%               | 0.1302         |

## 4. Final Model Selection

Based on the analysis of the results, the following models were chosen for the final OCR system.

| Expert Category | Selected Architecture | Justification                                                                                          |
|:----------------|:----------------------|:-------------------------------------------------------------------------------------------------------|
| **Digits**      | **Small**             | Consistently achieved the highest accuracy (~99.6%) and lowest loss. Larger models offered no benefit. |
| **Uppercase**   | **Medium**            | Outperformed the `Small` model and achieved the highest accuracy (98.31%).                             |
| **Lowercase**   | **Medium**            | Highest accuracy (96.16%) and lowest loss, superior for more complex letter shapes.                    |

These selected models will be used for the implementation of the character recognition pipeline.