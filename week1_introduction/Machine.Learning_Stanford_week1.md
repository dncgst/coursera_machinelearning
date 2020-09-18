COURSERA - Machine Learning - Week 1
====================================

## Introduction

Machine Learning (ML) definitions:
  > "Field of study that gives computers the ability to learn without being explicitly programmed" - Arthur Samuel, 1959

  > "A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E" - Tom Mitchell, 1998

ML algorithms:
  - Supervized Learning
  - Unsuprvized Learning
  - Reinforcement Learning
  - others...

Supervized Learning:
  - we give to the algorithm a dataset in which the "right answers" are given (input-output mapping).
  - Regression problem: predict continuous value output
  - Classification problem: predict a discrete value output
  - Support Vector Machine algorithm deals this an infinite number of variables

Unsuprvized Learning:
  - we give to the algorithm a dataset in which no "right answers" are given. With unsupervised learning there is no feedback based on the prediction results.
  - Clustering algorithm: find structures in the dataset (Google news, genome, organize computing clusters, social network analysis, market segmentation, astronomical data analysis)
  - Non-clustering algorithm: find structure in a chaotic environment (cocktail party problem)

## Linear Regression with One Variable

Training set:
  - $m$: number(#) of training examples
  - $x$: input (independent/predictor/explanatory/regressor/regular/...) variable
  - $y$: output (dependent/predicted/explained/response/target/...) variable
  - $(x,y)$: denote a single training example
  - $(x^{(i)}, y^{(i)})$: i training example
  - $X$: space of input values (e.g., Real numbers)
  - $Y$: space of output values

Training set -> Learning algorithm -> output the function h (hypothesis) that maps from x to y (input-output mapping)

How to represent h?
  In linear regression with 1 variable (univariate linear regression),
  - $h(x) = \theta_0 + \theta_1x$
  - $\theta_0$, $\theta_1$: parameters of the function

  In statistics: $y = \alpha + \beta(x)$, with $\beta$=slope and $\alpha$=intercept

Cost function (measures the accuracy of h):

  Choose $\theta_0$ and $\theta_1$ so that $h(x)$ is close to $y$ for our training example $(x,y)$ -> minimize on $\theta_0$ and $\theta_1$

  "Squared error function", or "mean squared error" $(J(\theta_0, \theta_1))$:

  $J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=1}^{m}(h(x^i)-y^i)^2$

  The square cost function is one of many cost function, but commonly used in linear regression: OSL - Ordinary Least Square (squared residual)
  The cost function for linear regression is always going to be a bow-shaped function, or a convex quadratic function! the local minimum (optimum) = global optimum

Gradient descent: algorithm to minimize the cost function J(th_0, th_1)
  Outline: start with theta_0, theta_1 values, repeat simultaneously update of th_0, th_1 values to reduce J() until getting at the local minimum.

  - := assignment
  - alpha: learning rate - if it is too large it may fail to converge to the local minimum or even diverge. But no need to decrease alpha over time, because the derivative term will be smaller and smaller as gradient descent approaches the local minimum

"Batch" Gradient Descent: each step of the gradient descent uses all the training examples

## Linear Algebra Review
