# Tomato Leaf Disease Classification

MACHINE LEARNING | IMAGE CLASSIFICATION | PYTHON ‧ CNN

A machine learning project that uses Convolutional Neural Networks (CNNs) to classify tomato leaf images into four categories: Bacterial Spot, Early Blight, Healthy, and Septoria Leaf Spot. The project compares four CNN architectures using 80:20 and 70:30 train-test split configurations and evaluates their predictions on external images.

## 01. Project Overview

### Project Description

Tomato Leaf Disease Classification is an image classification project designed to identify tomato leaf conditions from visual features using CNN models. The project evaluates how different CNN architectures perform under two dataset splitting configurations.

### Objectives

- Develop four CNN architectures for tomato leaf classification.
- Compare model accuracy using 80:20 and 70:30 train-test splits.
- Evaluate model predictions on external images.
- Analyze classification errors and prediction confidence.

### Problem Statement

Tomato leaves can exhibit visual symptoms associated with different diseases, making accurate identification important for crop monitoring. Image classification using CNNs can help recognize patterns in leaf images and provide preliminary classification results.

### Dataset

The dataset was obtained from Kaggle:

[Tomato Leaf Dataset — Kaggle](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)

A total of 800 images were selected, with 200 images for each class.

| Class | Number of Images |
|---|---:|
| Bacterial Spot | 200 |
| Early Blight | 200 |
| Healthy | 200 |
| Septoria Leaf Spot | 200 |
| **Total** | **800** |

### Solution

Four CNN architectures were trained and evaluated to classify images into the following categories:

- `Tomato___Bacterial_spot`
- `Tomato___Early_blight`
- `Tomato___healthy`
- `Tomato___Septoria_leaf_spot`

The project compares model accuracy across two train-test split configurations and uses six external images to examine prediction behavior beyond the dataset split.

## 02. Tech Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow, Keras |
| Image Processing | OpenCV |
| Data Processing | NumPy, Pandas |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning Utilities | Scikit-learn |
| Development Environment | Jupyter Notebook |

### Model Configuration

| Parameter | Configuration |
|---|---|
| Model Type | Convolutional Neural Network (CNN) |
| Number of Models | 4 |
| Input Image Size | 128 × 128 pixels |
| Number of Classes | 4 |
| Epochs | 20 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Output Activation | Softmax |

### CNN Architectures

- **Model 1:** Two convolutional layers with 32 and 64 filters, followed by dense layers.
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 126, 126, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 63, 63, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 63, 63, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 61, 61, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 30, 30, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 30, 30, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 57600)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │     7,372,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 4)              │           516 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
- **Model 2:** Two convolutional layers with 32 and 64 filters using 5 × 5 kernels, followed by a larger dense layer.
- **Model 3:** Two convolutional layers with 64 and 128 filters, followed by a larger dense layer.
- **Model 4:** Three convolutional layers with 32, 64, and 128 filters.

The architectures differ in convolutional kernel sizes, filter counts, network depth, and dense-layer configurations.

## 03. Screenshots / Results

### Machine Learning Workflow

1. **Dataset Collection:** Select 800 tomato leaf images across four classes.
2. **Image Preprocessing:** Resize images to 128 × 128 pixels and normalize pixel values.
3. **Dataset Splitting:** Prepare experiments using 80:20 and 70:30 train-test split configurations.
4. **Model Training:** Train four CNN architectures for 20 epochs.
5. **Model Evaluation:** Compare classification accuracy for each experiment.
6. **External Image Testing:** Test six separate images and compare predicted classes, confidence scores, and correctness.
7. **Results Visualization:** Display model accuracy comparisons and export external prediction results to CSV.

### Experiment 1 — 80:20 Train-Test Split

| Model | Accuracy |
|---|---:|
| Model 1 | **90.62%** |
| Model 2 | 80.00% |
| Model 3 | 84.38% |
| Model 4 | 86.25% |

**Best-performing model:** Model 1, with an accuracy of 90.62%.

### Experiment 2 — 70:30 Train-Test Split

| Model | Accuracy |
|---|---:|
| Model 1 | **84.17%** |
| Model 2 | 66.67% |
| Model 3 | 81.25% |
| Model 4 | 75.83% |

**Best-performing model:** Model 1, with an accuracy of 84.17%.

### Comparison Between Experiments

| Model | 80:20 Split | 70:30 Split | Difference |
|---|---:|---:|---:|
| Model 1 | 90.62% | 84.17% | +6.45 percentage points |
| Model 2 | 80.00% | 66.67% | +13.33 percentage points |
| Model 3 | 84.38% | 81.25% | +3.13 percentage points |
| Model 4 | 86.25% | 75.83% | +10.42 percentage points |

The 80:20 experiment achieved higher reported accuracy for all four models. Model 1 achieved the highest accuracy in both experiments.

### External Image Testing Results

Six external images were evaluated using each trained model. These results are separate from the model accuracy reported for the dataset test split.

**Experiment 1 — 80:20**

| Model | Correct | Incorrect | External Accuracy |
|---|---:|---:|---:|
| Model 1 | 5 | 1 | 83.33% |
| Model 2 | 5 | 1 | 83.33% |
| Model 3 | 4 | 2 | 66.67% |
| Model 4 | 4 | 2 | 66.67% |

**Experiment 2 — 70:30**

| Model | Correct | Incorrect | External Accuracy |
|---|---:|---:|---:|
| Model 1 | 4 | 2 | 66.67% |
| Model 2 | 4 | 2 | 66.67% |
| Model 3 | 4 | 2 | 66.67% |
| Model 4 | 4 | 2 | 66.67% |

### Insights

- **Model 1 performed best on the internal test split** in both experiments.
- The 80:20 configuration produced higher reported test accuracy across all four models.
- External image testing revealed classification errors involving Early Blight, Bacterial Spot, and Septoria Leaf Spot.
- High prediction confidence did not always correspond to a correct prediction, demonstrating that confidence scores should be interpreted alongside classification correctness.
- Internal test accuracy and external image accuracy differed, highlighting the importance of evaluating models on images beyond the original dataset split.

### Usefulness and Practical Application

The project demonstrates how CNNs can be used to classify tomato leaf images and compare alternative model architectures. Its results can support experimentation with image-based crop monitoring systems and provide a starting point for developing tools that assist users in identifying possible tomato leaf conditions.

The models are experimental and should not be treated as a definitive plant disease diagnosis system. Broader testing on diverse images is necessary before practical deployment.

### Generated Results

The external image testing module generates:

- `external_image_predictions.csv` — detailed predictions, confidence scores, and correctness for each image and model.
- `external_image_model_summary.csv` — correct predictions, incorrect predictions, and external accuracy for each model.

## 04. Installation & Usage

### Prerequisites

- Python environment compatible with the installed TensorFlow version.
- Jupyter Notebook or JupyterLab.
- The tomato leaf dataset downloaded from Kaggle.
- Six external test images stored in the `uji/` directory.

### Project Structure

```text
ml-TomatoLeafDiseaseClassification/
├── Percobaan 1_val.ipynb
├── Percobaan 2_val.ipynb
├── content/
│   └── dadata/
│       ├── Tomato___Bacterial_spot/
│       ├── Tomato___Early_blight/
│       ├── Tomato___healthy/
│       └── Tomato___Septoria_leaf_spot/
├── uji/
│   ├── bacterial_spot.jpg
│   ├── early_blight_1.jpg
│   ├── early_blight_2.jpg
│   ├── healthy.jpg
│   ├── septoria_leaf_spot_1.jpg
│   └── septoria_leaf_spot_2.jpg
└── README.md
```

Adjust the paths and notebook filenames if your repository uses a different structure.

### Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ml-TomatoLeafDiseaseClassification.git
cd ml-TomatoLeafDiseaseClassification
```

Install the required dependencies:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Running the Project

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf).
2. Extract the selected dataset images into the configured dataset directory.
3. Place the six external images inside the `uji/` directory.
4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `Percobaan 1_val.ipynb` to run the 80:20 experiment.
6. Open `Percobaan 2_val.ipynb` to run the 70:30 experiment.
7. Execute the notebook cells in order to preprocess the dataset, train the models, and evaluate their accuracy.
8. Run the external image testing section and choose either single-image testing or batch testing.
9. Review the generated visualizations, prediction results, and CSV files.

### Notes

- The notebooks train the models when their training cells are executed.
- Reported accuracy may vary depending on the dataset, data split, environment, and training configuration.
- The external evaluation uses six images and is intended as a limited supplementary evaluation.
- Update `YOUR_USERNAME` in the Git clone command with your GitHub username before using it.
