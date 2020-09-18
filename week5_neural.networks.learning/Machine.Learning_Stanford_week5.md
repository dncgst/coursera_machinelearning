COURSERA - Machine Learning - Week 5
====================================

## Neural Networks: Learning

### Cost Function

Notation:

* $L$ = total num. of layers in network
* $s_l$ = no. of units (without bias unit) in layer l
* $s_L$ or $K$= no. of units in the last layer. In binary classification problem, $s_L=K=1$. In multi-class classification, of course, $K \ge 3$.

The cost function is a generalization of the regularized cost function for logistic Regression

$J(\theta) = -\frac{1}{m}[\sum_{i=1}^{m} y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$

$+ \frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2]$

Thus, the cost function for NN is:

$J(\Theta) = -\frac{1}{m}[\sum_{i=1}^{m}\sum_{k=1}^{K} y_k^{(i)}\log(h_\Theta(x^{(i)}))_k + (1-y_k^{(i)})\log(1-h_\Theta(x^{(i)})_k)]$

$+ \frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_l+1}(\theta_{ji}^{l})^2]$

### Backpropagation

"Backpropagation" is neural-network terminology for minimizing our cost function, just like what we were doing with gradient descent in logistic and linear regression. We need to minimize $J(\Theta)$. In order to use either gradient descent or one of the advance optimization algorithms, we need to compute $J(\Theta)$ (see function above) and the partial derivative terms $\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)$.

First, compute the error of node j in layer l $\delta_j^{(l)}$

For example, for L=4

$\delta_j^{(4)} = a_j^{(4)}-y_j$

The vectorize implementation is:

* $\delta^{(4)} = a^{(4)}-y$
* $\delta^{(3)} = (\Theta^{(3)})^T\delta^{(4)}.*g'(z^{(3)})$
* $\delta^{(2)} = (\Theta^{(2)})^T\delta^{(3)}.*g'(z^{(2)})$

Thus, partial derivative result to be equal to:

$\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta) = a_j^{(l)}\delta_i^{(l+1)}$

(ignoring the regulariztion term $\lambda$)

#### Backpropagation algorithm

Given a training set ${(x^{(1)},y^{(1)}),\dots,(x^{(m)},y^{(m)})}$

* set $\Delta_{ij}^{(l)}=0$ (for all $l,i,j$)
* for i = 1:m
  - set $a^{(1)}=x{(i)}$
  - perform forward propagation to compute $a^{(l)}$ for $l=2,3,\dots,L$
  - using $j^{(i)}$, compute $\delta^{(L)}=a^{(L)}-y^{i}$
  - compute $\delta^{(L-1)},\delta^{(L-2)},\dots,\delta^{(2)}$
  - update $\Delta_{ij}^{l}:=\Delta_{ij}^{l}+a_j^{(l)}\delta_i^{(l+1)}$ with its vectorized form $\Delta^{(l)}:=\Delta^{(l)}+\delta^{(l+1)}(a^{(l)})^T$
* $D_{ij}^{(l)}:=\frac{1}{m}\Delta_{ij}^{(l)}+\lambda\Theta_{ij}^{(l)}$ if $j\ne0$
* $D_{ij}^{(l)}:=\frac{1}{m}\Delta_{ij}^{(l)}$ if $j=0$

Finally, $D_{ij}^{(l)} = \frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta) = a_j^{(l)}\delta_i^{(l+1)}$ (the partial derivative of the cost function, to be used in gradient descent or in other advanced optimization algorithms).

### Unrolling parameters

Given $s_1=10$, $s_2=10$, $s_3=1$,

$\Theta^{(1)}$ is a 10x11 matrix, $\Theta^{(2)}$ is a 10x11 matrix, $\Theta^{(3)}$ is a 1x11 matrix

$D^{(1)}$ is a 10x11 matrix, $D^{(2)}$ is a 10x11 matrix, $D^{(3)}$ is a 1x11 matrix

From matrices to vectors:

    thetaVec = [Theta1(:); Theta2(:); Theta3(:)];
    DVec = [D1(:); D2(:); D3(:)];

From vectors to matrices:

    Theta1 = reshape(thetaVec(1:110),10,11);
    Theta2 = reshape(thetaVec(111:220),10,11);
    Theta3 = reshape(thetaVec(221:231),1,11);

So, in order to use advanced optimization algorithms, such as `fminunc`, we first need to compute a function that returns the cost function and the derivatives:

    function [jVal, gradient] = costFunction(theta)
    ...

and than pass it to the advanced function:

    optTheta = fminunc(@costFunction, initialTheta, options)

So...

1. Unroll to get initialTheta to pass to fminunc:



2. Implement the costFunction:

        function [jVal, gradientVec] = costFunction (thetaVec)

        * from thetaVec, get Theta^1, Theta^2, Theta^3
        * Use FP, BP to compute D^1, D^2, D^3 and J(Theta)
        * Unroll D^1, D^2, D^3 to get gradientVec

## Gradient checking

Gradient checking will assure that our backpropagation works as intended. We can approximate the derivative of our cost function with:

$\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta) \approx \frac{J(\Theta+\epsilon)-J(\Theta-\epsilon)}{2\epsilon}$

A small value for ϵ{\epsilon}ϵ (epsilon) such as $\epsilon = 10^{-4}$, guarantees that the math works out properly. If the value for $\epsilon$ is too small, we can end up with numerical problems.

    n = length(theta)
    EPSILON = 10e-4

    for i = 1:n,
        thetaPlus = theta;
        thetaPlus(i) = thetaPlus(i) + EPSILON;
        thetaMinus = theta;
        thetaMinus(i) = thetaMinus(i) - EPSILON;
        gradApprox(i) = (J(thetaPlus)-J(thetaMinus))/2*EPSILON;
    end;

NOTE: Once you have verified that your backpropagation algorithm is correct, you don't need to compute gradApprox again. The code to compute gradApprox can be very slow.

## Random initialization: Symmetry breaking

Initialize each $\Theta_{ij}^{(l)}$ to a random value in $[-\epsilon,\epsilon]$

If the dimensions of Theta1 is 10x11 and Theta2 is 1x11:

    Theta1 = rand(10,11)*(2*INIT_EPSILON)-INIT_EPSILON;
    Theta2 = rand(1,11)*(2*INIT_EPSILON)-INIT_EPSILON;

## Summary

1. Pick a network architecture - choose the layout of your neural network, including how many hidden units in each layer and how many layers in total you want to have. Number of input units = dimension of features $x^{(i)}$. Number of output units = number of classes (in a multiclass classification problem). Number of hidden units per layer = usually more the better (if you have more than 1 hidden layer, then it is recommended that you have the same number of units in every hidden layer).
2. Randomly initialize the weights.
3. Implement forward propagation to get $h_\Theta(x^{(i)})$ for any $x^{(i)}$
4. Implement the cost function
5. Implement backpropagation to compute partial derivatives
6. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
7. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.
8. 
