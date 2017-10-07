function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sum=0;
for i=1:m
	sum=sum + ((-y(i,1)*log(sigmoid(X(i,:)*theta))) - ((1-y(i,1))*log(1-sigmoid(X(i,:)*theta))));
endfor
gamma=(1/m)*sum;	

% adding regularization term for cost function
a=0;
for i=2:size(theta)
	a=a+(theta(i,1)^2);
endfor
J=gamma + (lambda/(2*m))*a;



temp = sigmoid(X * theta);
temp1 = temp - y;
beta = temp1' * X;
g = (1/m)*beta';

% adding regularization term for gradient descent
for i=2:size(theta)
	g(i,1)=g(i,1)+(lambda/m)*theta(i,1); 
end
grad=g;

% =============================================================

end
