COURSERA - Machine Learning - Week 3
====================================

## Logistic Regression

> Logistic regression is a method for classifying data into discrete outcomes. For example, we might use logistic regression to classify an email as spam or not spam.

### Classification and Representation

Applying linear regression to a classification problem isn't a great idea. With y being a discrete value (either 0 or 1), linear regression

$h(x) = \theta^T x$

can output results $h(x)<0$ or $h(x)>1$. Logistic regression output instead is always $0\le h(x)\ge1$. Logistic regression is actually a classification algorithm!

Binary classification problem:

$y \in \{0,1\}$. 0 is also called the negative class, and 1 the positive class, and they are sometimes also denoted by the symbols “-” and “+”. Given $x(i)$, the corresponding $y(i)$ is also called the label for the training example.

Linear regression: $h_\theta(x) = \theta^T x$

Logistic regression: $h_\theta(x) = g(\theta^T x)$, where

$z=\theta^T x$

$g(z)=\frac{1}{1+e^{-z}}$ called Sigmoid or logistic function

Thus, $h_\theta(x) = g(\theta^T x) = \frac{1}{1+e^{-\theta^T x}}$

Interpretation of $h_\theta(x)$: $h_\theta(x)$ = estimated probability that $y=1$ on input $x$ (e.g. $h_\theta(x)=0.7$ 70% probability that y is positive)

  - $h_\theta(x) = P(y=1|x;\theta)$ "probability that y=1, given x, parameterized by $\theta$"
  - $P(y=0|x;\theta)=1-P(y=1|x;\theta)$ with $y∈\{0,1\}$

#### Linear decision boundary:

The line that separates the region where the hypothesis $h_\theta(x)$ predicts y=1 from the region where
$h_\theta(x)$ predicts y=0. It is a property of the hypothesis, including the parameters of the hypothesis. It is not a property of the dataset!

Suppose predict "y=1" if $h_\theta(x) \ge 0.5$

and "y=0" if $h_\theta(x) < 0.5$

Looking at the plot of $g(z)$, $g(z)\ge 0.5$ when $z\ge 0$

Thus, $h_\theta(x)\ge 0.5$ when $\theta^T x\ge 0$

and $h_\theta(x)< 0.5$ when $\theta^T x< 0$

#### Non-linear decision boundary:

e.g. $h_\theta(x)=g(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_1^2 + \theta_4 x_2^2)$

$\vec\theta = [\theta_0; \theta_1; \theta_2; \theta_3; \theta_4;]$ (column vector)

with for example $\vec\Theta = [-1; 0; 0; 1; 1]$

y=1 if $\theta^T x\ge 0$

  y=1 if $-1 + x_1^2 + x_2^2\ge 0$

  y=1 if $x_1^2 + x_2^2\ge 1$

Remember that $x_1^2 + x_2^2=1$ draws a **circle** of radius 1 and origin 0

With higher polynomial terms it is possible to get more complex decision boundary, such as ellipse or even more complex shapes!

[IDEA]: May be useful in palaeoanthropology for species classification, when traditional PCA and between group clustering is more fuzzy?

### Logistic Regression Model

The cost function $J(\theta)$ in linear regression:

$J(\theta) = \frac{1}{m}\sum_{i=1}^{m}\frac{1}{2}(h_\theta(x)-y)^2$

$cost(h_\theta(x)-y) = \frac{1}{2}(h_\theta(x)-y)^2$

would be a non-convex cost function if used in logistic regression!

Instead, for logistic regression the cost function $J(\theta)$ is:

$J(\theta) = \frac{1}{m}\sum_{i=1}^{m}cost(h_\theta(x)-y)$, with

  - $cost(h_\theta(x)-y) = -log(h_\theta(x))$ if y=1

    - $-log(h_\theta(x))$ if y=1

      it's a inverse logarithm curve with $cost=0$ if y=1, $h_\theta(x)=1$

      $cost\to \infty$ if y=1 and $h_\theta(x)\to 0$

  - $cost(h_\theta(x)-y) = -log(1-h_\theta(x))$ if y=0

    - $-log(1-h_\theta(x))$ if y=0

      $cost=0$ if y=0 and $h_\theta(x)=0$

      $cost\to \infty$ if y=0 and $h_\theta(x)\to 1$

$cost(h_\theta(x)-y)=0$ if $h_\theta(x)=y$

#### Logistic regression cost function

$cost(h_\theta(x)-y) = \left\{ -log(h_\theta(x)) \text{if y=1} \atop -log(1-h_\theta(x)) \text{if y=0} \right.$

can be written:

$cost(h_\theta(x)-y) = -y\log(h_\theta(x)) - (1-y)\log(1-h_\theta(x))$

Thus,

$J(\theta) = \frac{1}{m}\sum_{i=1}^{m}cost(h_\theta(x)-y)$

$= -\frac{1}{m}[\sum_{i=1}^{m} y\log(h_\theta(x)) + (1-y)\log(1-h_\theta(x))]$

NOTE that a vectorized implementation of $J(\theta)$ is:

$h=g(X\theta)$

$J(\theta) = \frac{1}{m}\big(-y^T\log(h)-(1-y)^T\log(1-h)\big)$

Apply Gradient Descent to minimaze $J(\theta)$

Repeat {

  $\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j}J(\theta)$

  (simultaneously update all $\theta_j$)

}

$\frac{\partial}{\partial\theta_j}J(\theta) = \frac{1}{m}\sum_{i=1}^m(h_\theta(x)-y)x_j$ (partial derivative term)

Repeat {

  $\theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^m(h_\theta(x)-y)x_j$

  (simultaneously update all $\theta_j$)

}

This equation is identical to the one used in linear regression!

BUT the definition of $h_\theta(x)$ is different

  - $= \theta^Tx$ in linear regression

  - $= \frac{1}{1+e^{-\theta^T x}}$ in logistic regression

To monitor gradient descent to make sure that it is converging, plot $J(\theta) = -\frac{1}{m}[\sum_{i=1}^{m} y\log(h_\theta(x)) + (1-y)\log(1-h_\theta(x))]$ as a function of the number of iterations and make sure $J(\theta)$ is decreasing on every iteration.

In order to implement gradient descent, updating all the paramethers value $\theta=[\theta_0; \theta_1; \theta_2; \dots \theta_n]$, instead of a for loop `for j = 0:n`, it is better to use a vectorized implementation!

NOTE that a vectorized implementation of gradient descent is:

$\theta := \theta-\frac{\alpha}{m}X^T\big(g(X\theta)-\vec y\big)$

Feature scaling also apply to logistic regression!

#### Optimization algorithms

- Gradient descent
- Conjugate gradient
- BFGS
- L-BFGS

Conjugate gradient, BFGS, L-BFGS

_Advantages_:
- no need to select $\alpha$
- often faster than gradient descent

_Disadvantages_:
- more complex

### Multiclass Classification

In multiclass classification problem $y$ can take on a small number of values $(y \in \{1,2,\dots,k\})$, not just a binary value.

One-vs-all classification: works turning the problem into $k$ separate binary classification problems, or $k+1$ if $y \in \{0,1,2,\dots,k\}$

Train a logistic regression classifier $h_\theta^{(i)}x$ for each class $i$ to predict the probability that $y=i$.

$h_\theta^{(i)}x = P(y=i|x;\theta)$

On a new input $x$, to make a prediction, pick the class $i$ that maximises max $h_\theta^{(i)}x$

## Regularization

Machine learning models need to generalize well to new examples that the model has not seen in practice. In this module, we introduce regularization, which helps prevent models from overfitting the training data.

### Solving the Problem of Overfitting

The problem of "**overfitting**" (or fitting an $h$ with high variance, as opposed to the problem of "**underfitting**", or fitting an highly biased $h$) comes when, if we have too many features, the learning hypothesis may fit very well, but fail to _generalize_ to new examples!

/!\ If we have a lot of features and very little training data then overfitting can become a problem!

Addressing overfitting:
1. reduce number of features
   - manually select what features to keep
   - model selection algorith (automatically select features)

   BUT selecting features also means disregard information that might be important

2. regularization
   - keep all the features, but reduce magnitude/values of parameters $\theta_j$

   WORKS WELL when we have a lot of features, each of which contributes a bit to predicting $y$

   Small values for parameters $\theta_0, \theta_1, \dots, \theta_n$ means:
    - simpler hypothesis
    - less prone to overfitting

#### Regularized linear regression

$J(\theta) = \frac{1}{2m}[\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2 + \lambda \sum_{i=1}^m\theta_j^2]$

Add this _regularization term_ ($\lambda \sum_{i=1}^m\theta_j^2$) at the end of the cost function to penalize every single parameter (but NOTE, not $\theta_0$ by convention).

The _regularization parameter_ $\lambda$ controls a trade off between two different goals: 1) fitting the training data well, and 2) keeping the parameters small.

/!\ extremely lare value of $lambda$ would underfit the hypothesis to the training set

##### Gradient descent

Repeat {

  $\theta_0 := \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}$

  $\theta_j := \theta_j - \alpha [\frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)} + \frac{\lambda}{m}\theta_j]$

  (simultaneously update $j \in \{1, 2,\dots, n\}$)

}

NOTE that $\theta_j := \theta_j - \alpha [\frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)} + \frac{\lambda}{m}\theta_j]$ can be written as:

$\theta_j := \theta_j(1-\alpha \frac{\lambda}{m}) -\alpha \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$

where $(1-\alpha \frac{\lambda}{m})$ is always a bit < 1!

##### Normal equation

The normal equation with no regularization $\theta = (X^TX)^{-1} X^T y$ becomes:

$\theta = (X^T X + \lambda L)^{-1} X^T y$

$L$ is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension (n+1)×(n+1). Intuitively, this is the identity matrix (though we are not including $x_0$, multiplied with a single real number λ.

NOTE with m (examples) < n (features) $(X^TX)$ will be **non-invertible**, or **singular**, or the matrix is said **degenerate**. It may be non-invertible if m=n.

`pinv` in Octave will do the computation, but it is not clear if it will give a good result, whereas `inv` will give an error.

Regularization, as lomg as $\lambda>0$, makes $(X^TX+\lambda [matrix])$ an **invertible** matrix

#### Regularized logistic regression

Cost function:

$J(\theta)= -\frac{1}{m}[\sum_{i=1}^{m} y\log(h_\theta(x)) + (1-y)\log(1-h_\theta(x))] + \\
\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$

##### Gradient descent

Repeat {

  $\theta_0 := \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}$

  $\theta_j := \theta_j - \alpha [\frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)} + \frac{\lambda}{m}\theta_j]$

  (simultaneously update $j \in \{1, 2,\dots, n\}$)

}

NOTE this algorith is similar to that for linear regression BUT in logistic regression the hypothesis $h_\theta(x)$ is different! $h_\theta(x)= \frac{1}{1+e^{-\theta^T x}}$

##### Advanced optimization

Write a `costFunction`:

    function [jVal, gradient] = costFunction(theta)
      jval = [code to compute J(theta)];
      gradient(1) = [code to compute the derivative J(theta0)];
      gradient(2) = [code to compute the derivative J(theta1)];
      gradient(n+1) = [code to compute the derivative J(thetan)];

where [code to compute J(theta)]:

$J(\theta)= -\frac{1}{m}[\sum_{i=1}^{m} y\log(h_\theta(x)) + (1-y)\log(1-h_\theta(x))] + \frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$

and [code to compute the derivative J(theta0)]:

$\frac{\partial}{\partial\theta_0}J(\theta) = \\
\frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}$

NOTE that $\theta_0$ is indexed $\theta_1$ in Octave!

and [code to compute the derivative J(theta_n)]:

$\frac{\partial}{\partial\theta_n}J(\theta) = \\
\big(\frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}\big) + \frac{\lambda}{m}\theta_n$

Then, `fminunc (@costFunction ...)`
