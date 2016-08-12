function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters,input_layer_size, num_labels)
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
          

  [J grad] = CostFunc(theta,input_layer_size, num_labels,X, y);
   theta=theta-alpha*grad;  
  J_history(iter)=J;
  fprintf('Iteration %d : %d\n', iter, J_history(iter))



end

end
