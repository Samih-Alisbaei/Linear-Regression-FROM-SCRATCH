# Linear Regression from Scratch in Python

A pure **NumPy** implementation of **Linear Regression** trained via **Gradient Descent** from scratch, without relying on high-level machine learning libraries like `scikit-learn`.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-013243)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project implements a foundational supervised learning algorithm—Linear Regression—built entirely from first principles. The goal of this project is to provide a clear, transparent look into how parameter optimization works under the hood using vectorization and gradient descent.

### Key Features
* **Zero ML Dependencies:** Built strictly using **NumPy** for linear algebra operations and **Matplotlib** for visualization.
* **Gradient Descent Optimization:** Iterative parameter update step-by-step.
* **Loss Tracking:** Monitors Mean Squared Error (MSE) over iterations to track convergence.

---

## 📐 Mathematical Formulation

### 1. Hypothesis Function
The hypothesis predicts a target value $\hat{y}$ given input feature $x$:

$$h_\theta(x) = w \cdot x + b$$

Where:
* $w$ = Weight parameter (slope)
* $b$ = Bias parameter (intercept)

### 2. Cost Function (Mean Squared Error)
We measure the error between predictions $\hat{y}$ and true values $y$ using MSE:

$$J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (w x_i + b))^2$$

### 3. Gradient Descent Updates
At each iteration, parameters are updated in the direction of steepest descent:

$$\frac{\partial J}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i)$$

$$\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

Parameter update rule:
$$w := w - \alpha \frac{\partial J}{\partial w}$$
$$b := b - \alpha \frac{\partial J}{\partial b}$$

*(where $\alpha$ is the learning rate)*

---

## 📊 Results & Visualizations

After running gradient descent for $100$ iterations with a learning rate of $\alpha = 0.01$, the model successfully learns the underlying linear relationship:

| Linear Fit Line | Loss Curve |
| :---: | :---: |
| ![Fitted Line](assets/fit_plot.png) | ![Cost Curve](assets/loss_plot.png) |

*(Note: Replace the image paths above with saved plots from your project, e.g., `plt.savefig('assets/fit_plot.png')`)*

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed. You can install the required dependencies using:

```bash
pip install -r requirements.txt
