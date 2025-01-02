# Multi-Layer Perceptron (MLP) Model Implementation

## Project Overview

This repository contains a minimal implementation of a Multi-Layer Perceptron (MLP) model built entirely from scratch using NumPy for all linear algebra operations. The implementation is inspired by Keras and serves as an educational project from our school, focusing on fundamental deep learning concepts.

---

## Key Features

### Type Safety and Modularity

- **Generic Type Annotations**: The code is fully written using Python's generic typing, ensuring clarity and type safety.

### Layer Architecture

- **Input Layer**: Accepts and standardizes input data.
- **Dense Layers**: Configurable activation functions for complex transformations.
- **Dropout Layers**: Helps prevent overfitting by randomly deactivating neurons during training.

### Optimizers

- **NAG (Nesterov Accelerated Gradient)**: Accelerates gradient descent using momentum.
- **RMSprop**: Adapts learning rates for each parameter.
- **Adam**: Combines momentum and adaptive learning rates for robust optimization.

### Activation Functions

- **Linear**: Identity mapping of inputs.
- **ReLU**: Introduces non-linearity and prevents vanishing gradients.
- **Sigmoid**: Maps values to the range (0, 1) for binary tasks (numerically stable).
- **Softmax**: Converts logits to probabilities for multi-class classification (numerically stable).

---

## What I Learned

Through this project, I gained the following insights and skills:

- **Deep Learning Foundations**: Understanding how MLPs are structured and trained from the ground up.
- **Optimization Techniques**: Practical implementation of advanced optimizers like Adam and NAG.
- **Numerical Stability**: Learning techniques to prevent overflow/underflow in critical operations such as softmax and sigmoid.
- **Coding Best Practices**: Writing modular, type-safe, and maintainable code using Python typing.
- **Debugging and Profiling**: Identifying and fixing issues in complex numerical computations.
- **Model Evaluation**: Analyzing model performance through metrics like accuracy, precision, recall, and F1-score.

---

## Public Methods

### Model Methods

1. **compile**

   - Purpose: Prepares the model for training by setting the loss function and optimizer.
   - Parameters:
     - `loss` (str): Name of the loss function (`binarycrossentropy`, `crossentropy`, or `mse`).
     - `optimizer` (str): Name of the optimizer (`gd`, `nag`, `rmsprop`, `adam`).
     - `lr` (float): Learning rate. Default is 0.001.
     - `beta1`, `beta2` (float): Hyperparameters for optimizers like Adam. Defaults are 0.9 and 0.99.
   - Example:
     ```python
     model.compile(loss='binarycrossentropy', optimizer='adam', lr=0.001)
     ```

2. **fit**

   - Purpose: Trains the model on the provided dataset.
   - Parameters:
     - `x`, `y` (ndarray): Training data and labels.
     - `batch_size` (int): Number of samples per batch. Default is 100.
     - `epochs` (int): Number of training epochs. Default is 13.
     - `val_split` (float): Fraction of training data to use as validation. Default is 0.2.
     - `early_stop` (bool): Enable early stopping. Default is False.
     - `early_stop_patience` (int): Number of epochs to wait for improvement. Default is 1.
   - Example:
     ```python
     history = model.fit(x_train, y_train, batch_size=32, epochs=10, val_split=0.1)
     ```

3. **predict**

   - Purpose: Makes predictions on the input data.
   - Parameters:
     - `x` (ndarray): Input data.
   - Returns: Predicted labels or probabilities.
   - Example:
     ```python
     predictions = model.predict(x_test)
     ```

4. **save** and **load**

   - Purpose: Save the model to disk and reload it.
   - Example:
     ```python
     model.save("my_model")
     model.load("my_model.npz")
     ```

---

## Losses, Optimizers, and Activations

### Loss Functions

- **Binary Crossentropy**: Suitable for binary classification tasks.
- **Crossentropy**: Ideal for multi-class classification.
- **Mean Squared Error (MSE)**: Used for regression tasks.

### Optimizers

- **NAG**: Accelerates gradient descent using momentum.
- **RMSprop**: Adapts learning rates for each parameter.
- **Adam**: Combines momentum and adaptive learning rates for robust optimization.

### Activation Functions

- **Softmax**: Converts logits to probabilities in multi-class tasks.
- **Sigmoid**: Maps values to the range (0, 1) for binary tasks.
- Implemented with numerical stability to prevent overflow/underflow issues.

---

## Early Stopping

Early stopping is implemented to prevent overfitting. Training halts if the validation loss does not improve for a specified number of epochs (`early_stop_patience`).

---

## Training History

The `fit` method returns a `History` object, which records:

- Training and validation loss per epoch
- Metrics (accuracy, precision, recall, F1-score)

Example:

```python
import matplotlib.pyplot as plt
plt.plot(history.loss, label='Training Loss')
plt.plot(history.val_loss, label='Validation Loss')
plt.legend()
plt.show()
```

---

## Future Work

- Expand to include additional optimizers like Adagrad or SGD.
- Support convolutional and recurrent neural networks.
- Add advanced regularization techniques like L1/L2 regularization.

---

## How to Use

1. **Initialize the Model**:

   ```python
   from mlp import Sequential
   from mlp.layers import Input, Dense, Dropout

   model = Sequential(
       Input(input_dim=10),
       Dense(output_dim=64, activation='relu'),
       Dropout(ratio=0.5),
       Dense(output_dim=1, activation='sigmoid')
   )
   ```

2. **Compile**:

   ```python
   model.compile(loss='binarycrossentropy', optimizer='adam', lr=0.001)
   ```

3. **Train**:

   ```python
   history = model.fit(x_train, y_train, batch_size=32, epochs=10, val_split=0.2)
   ```

4. **Evaluate**:

   ```python
   metrics = model.test(x_test, y_test)
   print(metrics)
   ```

---

## Contributions

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a clear description of your changes.

---

## Technical Details

- **Numerical Stability**: Softmax and sigmoid functions are implemented with techniques to prevent numerical overflow/underflow.
- **Design Choices**:
  - Modularity for flexibility.
  - Use of Python typing for code clarity and debugging.

---

