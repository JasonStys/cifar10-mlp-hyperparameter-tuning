# CIFAR-10 MLP Hyperparameter Tuning

This repository contains a CS478 deep learning assignment notebook that trains and tunes fully connected neural networks on the CIFAR-10 image classification dataset. The project uses Python, TensorFlow, Keras, NumPy, pandas, and Matplotlib to preprocess CIFAR-10 images, compare activation functions, experiment with hidden layer sizes, test optimizer and loss choices, and summarize validation accuracy results.

## Project Overview

The goal of this project is to explore how different neural network hyperparameters affect image classification performance. The notebook uses a multilayer perceptron, also called an MLP, rather than a convolutional neural network. Each 32 by 32 RGB image is flattened into a vector and passed through dense layers before a 10 class softmax output layer.

The assignment includes an initial round of manual experimentation followed by a cleaner tuning workflow that compares the baseline model against several modified models.

## Dataset

The project uses the CIFAR-10 dataset from Keras.

```text
Training images: 50,000
Validation images: 10,000
Image size: 32 by 32 pixels
Color channels: 3
Number of classes: 10
```

## CIFAR-10 Classes

```text
0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck
```

## Preprocessing

The notebook prepares the data by:

- Loading CIFAR-10 with Keras
- Inspecting the original training and validation shapes
- Displaying sample CIFAR-10 images with labels
- Converting image arrays to `float32`
- Normalizing pixel values from 0 to 255 into the range 0 to 1
- One hot encoding class labels for categorical classification

After preprocessing, the notebook shows:

```text
x_train shape: (50000, 32, 32, 3)
x_valid shape: (10000, 32, 32, 3)
y_train_cat shape: (50000, 10)
```

## Baseline Model

The baseline tuning model uses:

```text
Input: 32 by 32 by 3 image
Flatten layer
Dense hidden layer: 64 units
Activation: ReLU
Output layer: 10 units
Output activation: Softmax
Optimizer: Adam
Learning rate: 0.001
Loss: Categorical cross entropy
Epochs: 20
Batch size: 128
```

The baseline validation accuracy was:

```text
Baseline validation accuracy: 41.57 percent
```

## Hyperparameter Experiments

The final tuning workflow compares the baseline model against several changes.

| Experiment | Hidden Units | Extra Hidden Layer | Optimizer | Loss | Validation Accuracy |
| --- | ---: | --- | --- | --- | ---: |
| Baseline | 64 | No | Adam lr 0.001 | Categorical cross entropy | 41.57 percent |
| Hidden units 128 | 128 | No | Adam lr 0.001 | Categorical cross entropy | 46.12 percent |
| Hidden units 256 | 256 | No | Adam lr 0.001 | Categorical cross entropy | 47.51 percent |
| Hidden units 512 | 512 | No | Adam lr 0.001 | Categorical cross entropy | 48.41 percent |
| Extra hidden layer | 64 | Yes | Adam lr 0.001 | Categorical cross entropy | 47.56 percent |
| SGD optimizer | 64 | No | SGD lr 0.01 momentum 0.9 | Categorical cross entropy | 47.18 percent |
| Mean squared error loss | 64 | No | Adam lr 0.001 | Mean squared error | 39.11 percent |

## Best Result

The best final tuning result came from increasing the hidden layer to 512 units.

```text
Best model: ReLU MLP with 512 hidden units
Validation accuracy: 48.41 percent
Validation loss: 1.479989
Improvement over baseline: 6.84 percentage points
```

## Initial Experiment Notes

The notebook also includes earlier recorded experiment notes.

```text
ReLU: 43.01 percent
Softplus: 48.32 percent
Sigmoid: 48.22 percent

128 neurons: 51.37 percent
128 neurons with 2 layers: 56.37 percent
256 neurons: 50.55 percent
256 neurons with 2 layers: 53.09 percent
512 neurons: 51.48 percent
512 neurons with 2 layers: 51.53 percent
1024 neurons: 50.67 percent
```

These notes show that the assignment explored several activation functions and hidden layer sizes before the final organized tuning workflow.

## Key Takeaways

- Increasing hidden layer size improved validation accuracy in the final tuning workflow.
- The 512 unit ReLU model produced the best final validation accuracy.
- Adding a second hidden layer improved over the baseline, but did not outperform the 512 unit single hidden layer model.
- SGD with momentum improved over the Adam baseline in this run, but it did not beat the best hidden unit configuration.
- Mean squared error performed worse than categorical cross entropy for this multi class classification task.
- A fully connected MLP can classify CIFAR-10 images, but its performance is limited because it does not directly model spatial image features like a CNN.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- pandas
- Matplotlib
- Jupyter Notebook
- Google Colab

## Repository Structure

```text
.
├── Assignment_2 (1).ipynb
├── Assignment-2.pdf
├── README.md
└── requirements.txt
```

## How to Run

Clone the repository.

```bash
git clone https://github.com/your-username/cifar10-mlp-hyperparameter-tuning.git
cd cifar10-mlp-hyperparameter-tuning
```

Install the required packages.

```bash
pip install tensorflow keras numpy pandas matplotlib notebook
```

Launch Jupyter Notebook.

```bash
jupyter notebook
```

Open the notebook file and run all cells from top to bottom.

## Skills Demonstrated

- CIFAR-10 image classification
- TensorFlow and Keras model development
- Fully connected neural network design
- Image normalization
- One hot label encoding
- Activation function comparison
- Hidden layer size tuning
- Optimizer comparison
- Loss function comparison
- Training and validation accuracy analysis
- pandas result table creation
- Matplotlib visualization
- Jupyter Notebook workflow

## Possible Future Improvements

- Convert the MLP into a convolutional neural network for stronger image classification performance
- Add confusion matrix evaluation
- Add per class precision, recall, and F1 scores
- Add dropout and batch normalization experiments
- Add early stopping and model checkpointing
- Compare Adam, SGD, RMSprop, and FTRL in one complete table
- Add data augmentation
- Save the best trained model for reuse
