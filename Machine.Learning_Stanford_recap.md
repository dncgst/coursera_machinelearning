COURSERA - Machine Learning - Recap
===================================

## Introduction

In general, any machine learning problem can be assigned to one of two broad classifications: Supervised learning and Unsupervised learning

In **supervised learning**, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output. Supervised learning problems are categorized into "regression" and "classification" problems. In a *regression problem*, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a *classification problem*, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

**Unsupervised learning** allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables. We can derive this structure by *clustering* the data based on relationships among the variables in the data.

Other ML algorithms include: Reinforcement learning, ...

## Simple linear regression (with one variable)

Linear regression predicts a real-valued output based on an input value.

$x^i$ denotes the “input” (explanatory, independent) variables; $y^i$ denotes the “output” or target (independent) variable that we are trying to predict. A pair $(x^{(i)}, y^{(i)})$ is called a training example; a list of $m$ training examples is called a training set. $X$ denotes the space of input values, and $Y$ denotes the space of output values. For example, $X = Y = ℝ$.

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis.

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a *regression problem*.

When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a *classification problem*.

### *Cost function*

We can measure the accuracy of our hypothesis function by using a cost function. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's. This function is otherwise called the "Squared error function", or "Mean squared error".

$J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}
^{m}(h_\theta(x_i)-y_i)^2$

The idea is to minimaze $\theta_0$ and $\theta_1$ so that $h_\theta(x)$ is close to $y$ for our training sample $(x,y)$.

In visual terms, our training data set is scattered on the x-y plane. We are trying to make a straight line (defined by $h_\theta(x)$) which passes through these scattered data points. Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least.

NOTE: See below why the vectorization form $h_\theta(x)=\theta^Tx$ is computed as: X*theta

#### OCTAVE implementation: `computeCost.m`

    function J = computeCost(X, y, theta)

    m = length(y);
    J = 0;

    h = X*theta;
    sqrErrors = (h-y).^2;
    J = 1/(2*m) * sum(sqrErrors);

    end

### *Gradient descent*

So we have our hypothesis function $h_\theta(x)$ and we have a way of measuring how well it fits into the data $J(\theta_0,\theta_1)$. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.

The gradient descent algorithm is:

repeat until convergence:

$\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1)$

where

$j=0,1$ represents the feature index number.

At each iteration j, one should simultaneously update the parameters:

$temp0 := \theta_0 - \alpha \frac{\partial}{\partial\theta_0}J(\theta_0,\theta_1)$

$temp1 := \theta_1 - \alpha \frac{\partial}{\partial\theta_1}J(\theta_0,\theta_1)$

$\theta_0 := temp0$

$\theta_1 := temp1$

We should adjust our parameter $\alpha$ (learning rate) to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value imply that our step size is wrong.

### *Gradient Descent For Linear Regression*

When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to:

repeat until convergence: {

$\theta_0 := \theta_0 - \alpha\frac{1}{m}\sum_{i=1} ^{m}(h_\theta(x_i)-y_i)$

$\theta_1 := \theta_1 - \alpha\frac{1}{m}\sum_{i=1} ^{m}((h_\theta(x_i)-y_i)x_i)$

}

The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

This method looks at every example in the entire training set on every step, and is called batch gradient descent. Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges (assuming the learning rate α is not too large) to the global minimum. Indeed, J is a convex quadratic function.

#### OCTAVE implementation: `gradientDescent.m`

    function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

    m = length(y);
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters
      h = X*theta;
      err = h-y;
      grad = alpha*(1/m*(X'*err));
      theta = theta - grad;
      J_history(iter) = computeCost(X, y, theta);
    end;

    end

## Linear algebra review

* Matrices are 2-dimensional arrays. A vector is a matrix with one column and many rows. So vectors are a subset of matrices
* $A_{ij}$ refers to the element in the ith row and jth column of matrix A
* A vector with 'n' rows is referred to as an 'n'-dimensional vector
* $v_i$ refers to the element in the ith row of the vector
* Matrices are usually denoted by uppercase names while vectors are lowercase
* "Scalar" means that an object is a single value, not a vector or matrix
* $\mathbb{R}$ refers to the set of scalar real numbers
* Addition and subtraction are element-wise. To add or subtract two matrices, their dimensions must be the same
* In scalar multiplication, we simply multiply every element by the scalar value
* In scalar division, we simply divide every element by the scalar value
* __An *m x n* matrix multiplied by an *n x 1* vector results in an *m x 1* vector__
* __An *m x n* matrix multiplied by an *n x o* matrix results in an *m x o* matrix.__ To multiply two matrices, the number of columns of the first matrix must equal the number of rows of the second matrix
* Matrices are not commutative: $A∗B \neq B∗A$
* Matrices are associative: $(A∗B)∗C = A∗(B∗C)$
* The identity matrix, when multiplied by any matrix of the same dimensions, results in the original matrix. It's just like multiplying numbers by 1. The identity matrix simply has 1's on the diagonal (upper left to lower right diagonal) and 0's elsewhere. Initialized in octave with `eye()` - e.g., `eye(2)` initialize a 2x2 identity matrix.
* When multiplying the identity matrix after some matrix (A∗I), the square identity matrix's dimension should match the other matrix's columns. When multiplying the identity matrix before some other matrix (I∗A), the square identity matrix's dimension should match the other matrix's rows.
* The inverse of a matrix A is denoted $A^{-1}$ and computed in octave with `pinv(A)`. Multiplying by the inverse results in the identity matrix. A non square matrix does not have an inverse matrix.
* __The transposition of a matrix is like rotating the matrix 90° in clockwise direction and then reversing it. $A_{ij}=A^{T}_{ji}$__ Computed in octave with `A'`

## Multivariate linear regression (with multiple explanatory variables)

Linear regression with multiple variables is also known as "multivariate linear regression".

* $m$ = the number of training examples
* $n$ = the number of features

The multivariable form of the hypothesis function is:

$h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+\dots+\theta_nx_n$

NOTE that for convenience reasons in this course we assume $x_{0}^{(i)}=1$. This allows us to do matrix operations with theta and x. Hence making the two vectors $\theta$ and $x^{(i)}$ match each other element-wise (that is, have the same number of elements: n+1).

A vectorization of our hypothesis function for one training example is:

$h_\theta(x)=\theta^Tx$

Since X (the training examples) is size (m x n) and theta is size (n x 1), you arrange the order of operators so the result (a vector 'h' containing all of the hypothesis values - one for each training example) is size (m x 1) -> You can calculate the hypothesis as a column vector of size (m x 1) with:

$h_\theta(X)=X\theta$

### *Cost function*

#### OCTAVE implementation: `computeCostMulti.m`

    function J = computeCostMulti(X, y, theta)

    m = length(y);
    J = 0;

    h = X*theta;
    err = h-y;
    J = 1/(2*m) * err' * err;

    end

### *Gradient descent*

The gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:

repeat until convergence: {

$\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1} ^{m}(h_\theta(x_i)-y_i) ⋅ x_j^{(i)}$

for $j := 0 \dots n$

}

NOTE: If $\alpha$ is too small: slow convergence. If $\alpha$ is too large: ￼may not decrease on every iteration and thus may not converge.

A vectorization of Gradient descent is:

$\theta := \theta-\alpha\delta$

where $\delta = \frac{1}{m}\sum_{i=1} ^{m}(h_\theta(x_i)-y_i) ⋅ x_j^{(i)}$

#### OCTAVE implementation: `gradientDescentMulti.m`

    function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

    m = length(y);
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters
      h = X*theta;
      err = h-y;
      grad = alpha*(1/m*(X'*err));
      theta = theta - grad;
      J_history(iter) = computeCostMulti(X, y, theta);
    end

    end

*Feature scaling* and *mean normalization* are techniques to to speed things up by modifying the ranges of our input variables so that they are all roughly the same (e.g., $−1 \leq x_{(i)} \leq 1$).
*Feature scaling* involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. *Mean normalization* involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero.

#### OCTAVE implementation: `featureNormalize.m`

    function [X_norm, mu, sigma] = featureNormalize(X)

    X_norm = X;
    mu = zeros(1, size(X, 2));
    sigma = zeros(1, size(X, 2));

    mu = mean(X);
    sigma = std(X);
    m = size(X,1);
    mu_matrix = ones(m, 1)*mu;
    sigma_matrix = ones(m, 1)*sigma;
    X_norm = (X-mu_matrix)./sigma_matrix;

    end

### *Polynomial regression*

We can improve our features and the form of our hypothesis function in a couple different ways. E.g., We can combine multiple features into one. For example, we can combine *x_1* and *x_2* into a new feature $x_3$ by taking $x_1⋅x_2$.

Or, we can change the behavior or curve of our hypothesis function by making it a quadratic, cubic or square root function (or any other form), if our hypothesis function need not be linear (a straight line) in order to fit the data well. For example, if our hypothesis function is $h_\theta(x)=\theta_0+\theta_1x_1$, then we can create additional features based on $x_1$: $h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_1^2$ (quadratic function); $h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_1^2+\theta_3x_1^3$ (cubic function)

NOTE: If you choose your features this way then feature scaling becomes very important!

### *Normal Equation*

Gradient descent gives one way of minimizing J. Let’s discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm. In the "Normal Equation" method, we will minimize J by explicitly taking its derivatives with respect to the θj ’s, and setting them to zero. This allows us to find the optimum theta without iteration. The normal equation formula is given below:

$\theta=(X^T X)^{-1} X^T y$

NOTE: There is no need to do feature scaling with the normal equation.

However, if we have a very large number of features n, the normal equation will be slow. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process (gradient descent).

If $X^T X$ is non invertible, the common causes might be having:
  * Redundant features, where two features are very closely related (i.e. they are linearly dependent)
  * Too many features (e.g. m ≤ n). In this case, delete some features or use "regularization" (to be explained in a later lesson).

NOTE: in octave we want to use the `pinv` function rather than `inv`. The `pinv` function will give you a value of $\theta$ even if $X^T X$ is not invertible.

#### OCTAVE implementation: `normalEqn.m`

    function [theta] = normalEqn(X, y)

    theta = zeros(size(X, 2), 1);

    theta = pinv(X'*X) * X' * y

    end

## Logistic regression (binary classification)

Logistic regression is a method for classifying data into discrete outcomes (*classification problem*). For example, we might use logistic regression to classify an email as spam or not spam.

Instead of our output vector y being a continuous range of values, it will only be 0 or 1. While the linear regression $h(x) = \theta^T x$ can output results $h(x)<0$ or $h(x)>1$, logistic regression output instead is always $0\le h(x)\ge1$.

Logistic regression: $h_\theta(x) = g(\theta^T x)$, where

$z=\theta^T x$

$g(z)=\frac{1}{1+e^{-z}}$ called Sigmoid or logistic function

$h_\theta$ will give us the probability that our output is 1. For example, $h_\theta(x)=0.7$ gives us the probability of 70% that our output is 1. Our probability that our prediction is 0 is just the complement of our probability that it is 1 (e.g. if probability that it is 1 is 70%, then the probability that it is 0 is 30%).

#### OCTAVE implementation: `sigmoid.m`

    function g = sigmoid(z)

    g = zeros(size(z));

    g = 1./(1+exp(-z));

    end

### *Decision boundary*

A linear decision boundary separates the region where the hypothesis $h_\theta(x)$ predicts y=1 from the region where
$h_\theta(x)$ predicts y=0. It is a property of the hypothesis, including the parameters of the hypothesis. It is not a property of the dataset!

It predicts "y=1" if $h_\theta(x) \ge 0.5$ and "y=0" if $h_\theta(x) < 0.5$.

Looking at the plot of $g(z)$,

$g(z)\ge 0.5$ when $z\ge 0$. Thus, $h_\theta(x)\ge 0.5$ when $\theta^T x\ge 0$

and $h_\theta(x)< 0.5$ when $\theta^T x< 0$

The input to the sigmoid function g(z) doesn't need to be linear (e.g., $\theta^TX$), and could be a function that describes a circle (e.g. $z=\theta_0 + \theta_1 x_1 + \theta_2 x_2^2$) or any shape to fit our data.

#### OCTAVE implementation: `predict.m`

    function p = predict(theta, X)

    m = size(X, 1);
    p = zeros(m, 1);

    h = sigmoid(X*theta);
    for i = 1:m
      if h(i) >= 0.5
      p(i) = 1;
      else
      p(i) = 0;
      endif
    end

    end

### *Cost function*

For logistic regression the cost function $J(\theta)$ is:

$J(\theta) = \frac{1}{m}\sum_{i=1}^{m}cost(h_\theta(x^{i})-y^{i})$ , with

$cost(h_\theta(x)-y) = \left\{ -log(h_\theta(x)) ...............\text{if y=1} \atop -log(1-h_\theta(x)) ..........\text{if y=0} \right.$


Thus, $cost(h_\theta(x)-y)=0$ if $h_\theta(x)=y$ >>> If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. On the other hand, if our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1.

In order to simplify it, we can compress our cost function's two conditional cases into one case:

$cost(h_\theta(x)-y) = -y\log(h_\theta(x)) - (1-y)\log(1-h_\theta(x))$

Thus,

$J(\theta) = -\frac{1}{m}[\sum_{i=1}^{m} y\log(h_\theta(x)) + (1-y)\log(1-h_\theta(x))]$

NOTE that a vectorized implementation of $J(\theta)$ is:

$h=g(X\theta)$

$J(\theta) = \frac{1}{m}\big(-y^T\log(h)-(1-y)^T\log(1-h)\big)$

#### OCTAVE implementation: `costFunction.m`

    function [J, grad] = costFunction(theta, X, y)

    m = length(y);
    J = 0;
    grad = zeros(size(theta));

    h = sigmoid(X*theta);
    J = 1/m*(
      (-y'*log(h))-
      ((1-y)'*log(1-h))
      );
    grad = 1/m*(
      X'*(h-y)
      );

    end

### *Gradient descent*

The algorithm is identical to the one we used in linear regression! BUT the definition of $h_\theta(x)$ is different!

NOTE that a vectorized implementation of gradient descent is:

$\theta := \theta-\frac{\alpha}{m}X^T\big(g(X\theta)-\vec y\big)$

Feature scaling also apply to logistic regression!

#### OCTAVE implementation:

"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize θ that can be used instead of gradient descent.

The built-in function called `fminunc` is an optimization solver that finds the minimum of an unconstrained function. For logistic regression, you want to optimize the cost function J(θ) with parameters $\theta$. Use `fminunc` to find the best parameters θ
for the logistic regression cost function, given a fixed dataset (of X and y values).

You need to supply J() and the partial derivative terms...

## Logistic regression (multiclass classification: one-vs-all)

For the classification of data into more than two categories. Instead of y = {0,1} we will expand our definition so that y = {0,1...n}.

One-vs-all classification works turning our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

#### OCTAVE implementation:

See below the regularized version `costFunctionReg.m`

## Regularization

Regularization is designed to address the problem of overfitting. High bias or underfitting is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting or high variance is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:

  1. Reduce the number of features
     a. Manually select which features to keep
     b. Use a model selection algorithm
  2. Regularization: Keep all the features, but reduce the parameters $\theta_j$

$J(\theta) = \frac{1}{2m}[\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2 + \lambda \sum_{i=1}^m\theta_j^2]$

Add this _regularization term_ ($\lambda \sum_{i=1}^m\theta_j^2$) at the end of the cost function to penalize every single parameter (but NOTE, not $\theta_0$ by convention).

The _regularization parameter_ $\lambda$ controls a trade off between two different goals: 1) fitting the training data well, and 2) keeping the parameters small.

/!\ extremely large value of $lambda$ would underfit the hypothesis to the training set

### *Regularized linear regression*

### *Regularized logistic regression*

#### OCTAVE implementation: `costFunctionReg.m`

    function [J, grad] = costFunctionReg(theta, X, y, lambda)

    m = length(y);
    J = 0;
    grad = zeros(size(theta));

    h = sigmoid(X*theta);

    theta(1) = 0;

    J = 1/m*(
      (-y'*log(h))-
      ((1-y)'*log(1-h))
      )+(
      (lambda/(2*m))*(theta'*theta)
      );

    grad = 1/m*(
      X'*(h-y)
      )+(
      (lambda/m)*theta
      );

    end

## Neural networks

NN are learning algorithms, like linear regression and logistic regression, used to learn complex non-linear hypotheses. Performing linear regression with a complex set of data with many features is very unwieldy (ss the number of our features increase, the number of quadratic or cubic features increase very rapidly and becomes quickly impractical).

In its simplest representation an artificial neuron has a Sigmoid or logistic _activation function_ $g(z)=\frac{1}{1+e^-z}$, where parameters $\theta$ are called _weights_.

In a neural networks, the _input_ layer is where we first input the features $x_0, x_1, x_2, \dots, x_n$. The middle _hidden_ layer is where the neural network $a_0, a_1, a_2, \dots, a_n$ is placed. There can be more than one hidden layer. The final _output_ layer is where a neuron output the computed value of $h_\Theta(x)$.

Notation:

- $a_i^{(j)}$: activation of unit $i$ in layer $j$
  - e.g. $a_1^{(2)}$ is the 1st unit of layer 2
      - $a_1^{(2)} =g(z) =g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3)$, where input layer 1 $\in {x_1, x_2, x_3}$ and $x_0$ is the bias feature always = 1.
- $a_i^{(j)}=g(z_i^{(j)})$
- $\Theta^{(j)}$: matrix of weights controlling function mapping from layer $j$ to layer $j+1$
  - If network has $s_j$ units in layer $j$, $s_{j+1}$ in layer $j+1$, then $\Theta^{(j)}$ will be of dimension $s_{j+1}\times(s_j+1)$

### *Forward propagation* - Vectorized implementation:

With $x$ = column of vector of $x_0, x_1, x_2, \dots, x_n$ at the 1st layer,

and $z^{(2)}$ = column of vector of $z_1^{(2)}, z_2^{(2)}, z_3^{(2)}, \dots, z_n^{(2)}$ at the 2nd layer,

at the 2nd layer we have:

$z^{(2)}=\Theta^{(1)}x$

$a^{(2)} = g(z^{(2)})$

Plus a a bias activation function $a_0^{(2)}=1$ after you computed $a^{(2)}$!

At the 3rd layer we have:

$z^{(3)}=\Theta^{(2)}a^{(2)}$

$h_\Theta(x)=a^{(3)}=g(z^{(3)})$

To generalize: $h_\Theta(x)=a^{(j+1)}=g(z^{(j+1)})$

#### OCTAVE implementation: `predict.m`

    function p = predict(Theta1, Theta2, X)

    m = size(X, 1);
    num_labels = size(Theta2, 1);

    p = zeros(size(X, 1), 1);

    % Add ones to the X data matrix -> X = m x n matrix -> a1 = m x n+1
    a1 = [ones(m, 1) X];

    % Multiply by Theta1 (j x n+1) and you have 'z2'
    z2 = a1*Theta1';

    % Compute the sigmoid() of 'z2', then add a column of 1's, and it becomes 'a2'
    aa = sigmoid(z2);
    a2 = [ones(size(aa,1), 1) aa];

    % Multiply by Theta2, compute the sigmoid() and it becomes 'a3'.
    z3 = a2*Theta2';
    a3 = sigmoid(z3);

    % Now use the max(a3, [], 2) function to return two vectors - one of the highest
    % value for each row, and one with its index. Ignore the highest values.
    % Keep the vector of the indexes where the highest values were found.
    % These are your predictions.

    [v,i] = max(a3, [], 2);

    p = i;

    end

### *Cost function* and *Backpropagation*

###
