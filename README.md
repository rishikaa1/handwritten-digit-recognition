# Handwritten Digit Recognition

This repository contains various machine learning models implemented for recognizing handwritten digits. The models range from traditional machine learning algorithms to more advanced deep learning approaches. The project is built using Python, Colab and Jupyter notebooks.

## Models Implemented

1. **Logistic Regression**: A classic linear model used for multi-class classification tasks. This model is trained on the MNIST dataset to recognize digits from 0 to 9.  
   - [Notebook](./logistic_regression.ipynb)

2. **Linear SVM**: A Support Vector Machine with a linear kernel, trained to classify handwritten digits.  
   - [Notebook](./svm_linear.ipynb)

3. **RBF Kernel SVM**: An advanced Support Vector Machine using the Radial Basis Function (RBF) kernel, which allows for non-linear decision boundaries to better fit the digit data.  
   - [Notebook](./rbf_kernel_svm.ipynb)

4. **Convolutional Neural Network (CNN)**: A deep learning model specifically designed for image recognition tasks, leveraging convolutional layers to extract spatial features from the images.  
   - [Notebook](./CNN.ipynb)

## Dataset

The models are trained on the **MNIST dataset**, which contains 70,000 grayscale images of handwritten digits (28x28 pixels each) divided into 60,000 training samples and 10,000 test samples. The goal is to classify the digits correctly.

## Project Report

A comprehensive report detailing the methodology, dataset, models, training processes, and evaluation results is available:
- [report.pdf](./report.pdf)

## Dependencies
The following libraries are required to run the notebooks:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `keras`

The dependencies can be installed using the provided `requirements.txt` file.

## Model Evaluation
Each model is evaluated based on their accuracy. The CNN model, being a deep learning approach, typically outperforms the traditional models, especially for complex digit patterns.

### Accuracies (rounded off to 2 digits)
- Logistic Regression: 91.45%
- Linear SVM: 92.00%
- RBF Kernel SVM: 96.57%
- CNN (without hidden layer): 92.48%; (with hidden layer): 97.35%

## Future Scope
- **Model Optimization:** Hyperparamter tuning for SVM and CNN models to improve performance.
- **Data Augmentation:** Implementing techniques such as rotation, scaling, and noise addition to enhance the training dataset for CNN.
- **Transfer Learning:** Exploring pre-trained models to further boost accuracy.

## About the Project
This project was built as part of the academic internship during the 4th semester of B.Tech in Computer Science and Engineering at Jorhat Engineering College.

## License
This project is licensed under the [Apache License Version 2.0](./LICENSE).
