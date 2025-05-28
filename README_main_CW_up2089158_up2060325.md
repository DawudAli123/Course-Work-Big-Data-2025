# M32895 Coursework - CIFAR-100 Image Classification Using CNN

## Project Overview

This submission for the M32895 Big Data Applications module demonstrates the development of a full machine learning pipeline for multi-class image classification using the CIFAR-100 dataset. The project is structured in alignment with the lecture notes provided during the course, but further extended to include enhanced model architectures, multi-model evaluation, and deeper performance insight.

The CIFAR-100 dataset contains 60,000 colour images sized 32x32 pixels, evenly distributed across 100 distinct object categories. These categories range from animals to vehicles, trees, and common objects. Each image is labelled with a single class. The dataset is loaded from the `tensorflow.keras.datasets` module and split into training, validation, and test sets.

---

## Code Breakdown and Justification

### 1. Importing Libraries

The notebook begins by importing essential Python libraries such as `numpy`, `pandas`, `matplotlib`, `seaborn`, `tensorflow`, and `sklearn`. These libraries are used for numerical operations, plotting, deep learning model development, and evaluation. The use of `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'` suppresses unnecessary TensorFlow logs for cleaner output during execution.

### 2. Data Loading and Initial Exploration

The CIFAR-100 dataset is loaded using the `cifar100.load_data()` function from Keras. The shapes of the training and test sets are printed to confirm the structure of the data. An example image is displayed using `matplotlib` to visually confirm the image size and content.

### 3. Data Integrity Check

A custom function `check_images()` is defined to confirm that all images are valid NumPy arrays with expected shapes and pixel values. Although CIFAR-100 is known to be well-formatted, this function mimics robust data practices and simulates real-world checks where data may be malformed.

### 4. Splitting into Train/Validation/Test

The dataset is split using `train_test_split()` to ensure we have a validation set distinct from the training data. This allows for unbiased performance assessment during training without peeking at the test set.

### 5. Class Name Mapping

A full list of class names for CIFAR-100 is defined. This is necessary for interpreting label outputs, generating readable confusion matrices, and plotting class-level predictions.

### 6. Label Distribution Analysis

A frequency count of each class in the train, validation, and test sets is generated using a helper function. This ensures label distribution is balanced after splitting. A `seaborn` barplot visualises this distribution, helping verify there is no data leakage or imbalance.

### 7. Image Normalisation

Images are scaled from their original pixel range (0–255) to the range 0–1 using `astype('float32') / 255.0`. This helps the model converge faster and avoids numerical instability during training.

### 8. One-Hot Encoding of Labels

The `to_categorical()` function converts class labels into one-hot encoded vectors. This is required for multi-class classification using softmax output and categorical cross-entropy loss.

---

## Model Development

### 9. Model 1 – Baseline CNN

This model features a simple CNN architecture with two convolutional layers followed by max pooling, a dense layer, and dropout. It serves as a minimal baseline for performance benchmarking.

### 10. Model 2 – CNN with Batch Normalization and Dropout

This model builds on Model 1 by adding batch normalization layers after convolutional blocks and increasing dropout. This stabilises learning and improves generalisation.

### 11. Model 3 – Deeper CNN

This third model increases the number of convolutional layers and filters. It is designed to test the hypothesis that greater model complexity will yield improved validation/test accuracy. This model performed the best in terms of accuracy.

---

## Model Fitting and Evaluation

Each model is trained separately using the same training and validation sets. The `EarlyStopping` callback monitors validation loss to prevent overfitting. After training, performance metrics including loss and accuracy are plotted using `matplotlib` for each model.

A final summary DataFrame compares validation and test accuracies across all three models to clearly highlight improvements through architectural changes.

---

## Performance Metrics

To understand each model’s classification performance:

- Confusion matrices are generated using `sklearn.metrics.confusion_matrix` to show class-level prediction performance.
- Classification reports with precision, recall, and F1-score are printed for each model using `classification_report()`.
- A function `clf_performance()` calls these metrics on the train, validation, and test sets for each model in a standardised format.

---

## Individual Prediction Analysis

A single test image is selected and visualised along with its one-hot label. Each model is used to predict the class of the image, and the predicted class index is mapped back to a human-readable class name.

The probability outputs from the softmax layer are used to generate a probability distribution across all 100 classes using Plotly. This visualisation helps illustrate model confidence and shows whether the correct class had high probability, even if not the top prediction.

---

## Tools Used

- **TensorFlow/Keras**: For model building and training

- **Matplotlib / Seaborn / Plotly**: For visualisation

- **Scikit-learn**: For performance metrics

- **NumPy / pandas**: For data manipulation and formatting

---

## Results Summary

Performance across all three models showed clear incremental improvements. Model 1 performed reasonably as a baseline. Model 2 benefited from improved regularisation. Model 3, the deeper CNN, achieved the best performance on both validation and test sets, confirming the positive impact of additional depth.

---

## Use of AI Assistance

Some sections of this notebook were written or refined with limited support from AI tools (ChatGPT) to improve code clarity and ensure Python syntax consistency—particularly in parts related to performance visualisation, probability interpretation, and formatting of classification metrics. All model architecture design, dataset processing, and final performance evaluation were independently developed, with AI used solely as a supplementary code refinement aid. The authors ensured full understanding and control over all outputs before inclusion.

Limited use of AI tools (ChatGPT) was employed during the development of this coursework to assist in the following specific areas:

- Model Evaluation Visualisation: Support was used to format loss/accuracy plots and clarify performance trends across epochs.

- Performance Summary Table: Assistance was provided to generate code for compiling and presenting the validation and test accuracies of the three models in a structured format.

- Confusion Matrix Formatting: AI support helped refine the display of multi-class confusion matrices and classification reports for clearer readability.

- Final Probability Bar Plot: Guidance was used to enhance the visualisation of class probabilities using Plotly, particularly for single-image prediction analysis.

---

## How to Run

The code is fully contained within a Jupyter Notebook. Simply execute the cells in sequence to load data, train models, and evaluate results. The script is designed to run in VS Code or JupyterLab environments with minimal setup.

---

## Authors

UP2089158 and UP2060325  
University of Portsmouth  
M32895 Big Data Applications Coursework (2025)
