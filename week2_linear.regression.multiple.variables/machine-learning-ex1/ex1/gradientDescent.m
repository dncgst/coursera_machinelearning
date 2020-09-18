function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
% X is a (mxn) matrix
% y is a (mx1) vector
% theta is a (nx1) column vector: fitting parameters theta = zeros(2, 1)
% theta is a column vector (2x1) with theta0=0 and theta1=0
% predictions of hypothesis on all m: prediction (h) = X*theta
% h=theta'*x is true only when theta and x are both column vectors
% h=X*theta is true when X is the whole matrix of training examples
% cost function J(theta): J = 1/(2*m) * sum((predictions-y).^2);

% for-loop method
%for j = 1:2
%derivative = 1/m*sum((X*theta-y)*X);
%theta(j) = theta(j) - (alpha * derivative);
%end

% vectorized method
h = X*theta; % (mx1) vector
err = h-y; % (mx1) vector
grad = alpha*(1/m*(X'*err));
theta = theta - grad;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
end