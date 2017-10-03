function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression

m = length(y); % number of training examples
predictions = X*theta;
sqErrors = (predictions-y).^2;
J = 1/(2*m)*sum(sqErrors);

end
