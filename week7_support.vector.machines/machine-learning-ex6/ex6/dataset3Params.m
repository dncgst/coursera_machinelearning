function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% 1x8 row vector of C and sigma values
C_list = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_list = [0.01 0.03 0.1 0.3 1 3 10 30];

% create a blank results matrix (64x3) - all possible pairs of values for C and σ = 8^2=64
% 1st column (C), second column (sigma), 3rd column (error on the cross validation set)
results = zeros(length(C_list) * length(sigma_list), 3);

row = 1;

for c = C_list % for each value c of C
  for s = sigma_list % for each value s of sigma
    
    % train using C_val (c) and sigma_val (s)
    model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s)); 

    % compute the validation set errors 'err_val'
    predictions = svmPredict(model, Xval);
    err_val = mean(double(predictions ~= yval));
    
    % save the results in the matrix
    results(row,:) = [c s err_val];
    row = row + 1;
  end
end

% use the min() function on the results matrix to find 
%   the C and sigma values that give the lowest validation error

[v i] = min(results(:,3));
C = results(i,1);
sigma = results(i,2);

% =========================================================================

end