function [J grad] = CostFunc(nn_params, ...
                             input_layer_size, ...
                             num_labels, ...
                             X, y)

Theta1 = reshape(nn_params(1:num_labels * (input_layer_size + 1)), ...
                 num_labels, (input_layer_size + 1));


m = size(X, 1);
         

J = 0;
Theta1_grad = zeros(size(Theta1));

X=[ones(m,1) X];

for i=1:m,
     Z2=Theta1*(X(i,:))';
     a2=sigmoid(Z2);
     
	 y1=[1:num_labels];
     %I am hardcoring it wen i write y1=[1:10];
	 y1=y1';
     y1=(y1==y(i));
     %y1=eye(num_labels);
	 A=log(a2);
     B=log(1-a2);
     D=(1-y1).* B;
	 C=y1 .* A;
     E= - C - D;
     F=sum(E);
     G=F/m;
     J=J+G;     
end	 

% Implementing backpropagation using Vectorization

a1=X;
Z2=X*Theta1';
a2=sigmoid(Z2);

y_matrix=y;
% d3=a3-y_matrix;
% d2=d3*Theta2(:,2:end) .* sigmoidGradient(Z2);
% Theta1_grad=d2'*a1;
% Theta2_grad=d3'*a2;

d2=a2-y_matrix;
Theta1_grad=d2'*a1;

Theta1_grad=Theta1_grad/m;



% =========================================================================

% Unroll gradients
grad = Theta1_grad;


end
