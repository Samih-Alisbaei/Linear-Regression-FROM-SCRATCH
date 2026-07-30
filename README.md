# Linear Regression from Scratch in Python

A simple NumPy implementation of Linear Regression trained with Gradient Descent, no scikit-learn or other ML libraries involved.

## Overview

I built this to really understand what's happening under the hood when a model "learns." Instead of calling `.fit()` on some library, everything here is done manually with vectorized NumPy operations — the hypothesis function, the cost function, and the gradient descent updates.

### Features

- No ML dependencies — just NumPy for the math and Matplotlib for plotting
- Gradient descent implemented step by step
- Tracks MSE loss over iterations so you can see it converge

## The Math

### Hypothesis Function

The model predicts `y_hat` from input `x` using:

```
h(x) = w * x + b
```

- `w` — weight (slope)
- `b` — bias (intercept)

### Cost Function (MSE)

```
J(w, b) = (1/n) * sum((y_i - (w*x_i + b))^2)
```

### Gradient Descent

Gradients:

```
dJ/dw = -(2/n) * sum(x_i * (y_i - y_hat_i))
dJ/db = -(2/n) * sum(y_i - y_hat_i)
```

Update rule (alpha = learning rate):

```
w = w - alpha * dJ/dw
b = b - alpha * dJ/db
```

## Results
As you can see, every iteration, the cost converges :


<img width="652" height="727" alt="image" src="https://github.com/user-attachments/assets/13107e72-155b-4b34-a1f9-52a1e742d6d8" />


Ran for 200 iterations at a learning rate of 0.01, and it picks up the underlying linear trend pretty well:

![Linear Fit Line](https://github.com/user-attachments/assets/12c559bc-bca1-46c4-95f0-07baae326d20)

