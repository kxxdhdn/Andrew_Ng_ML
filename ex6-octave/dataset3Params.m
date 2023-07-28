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
C_try = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_try = [0.01 0.03 0.1 0.3 1 3 10 30];
s = length(C_try);
t = length(sigma_try);
error_val = zeros(s, t);

for i=1:s
    for j=1:t
        display(['C = ', num2str(C_try(i)), '; sigma = ', num2str(sigma_try(j))]);
        model_train = svmTrain( X, y, C_try(i), @(x1, x2) gaussianKernel(x1, x2, sigma_try(j)) );
        predictions = svmPredict(model_train, Xval);
        error_val(i, j) = error_val(i, j) + mean(double(predictions ~= yval));
    end
end

## Unlike Matlab, Octave does not support all-dim min/max
[min_col, ind_row] = min(error_val, [], 1);
[min_all, icol] = min(min_col, [], 2);
irow = ind_row(icol);
C = C_try(irow);
sigma = sigma_try(icol);
display(['C = ', num2str(C_try(irow)), '; sigma = ', num2str(sigma_try(icol))]);

% =========================================================================

end
