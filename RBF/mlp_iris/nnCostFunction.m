function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X=[ones(m,1) X];

for i=1:m,
     Z2=Theta1*(X(i,:))';
     a2=sigmoid(Z2);
     a2=[1;a2];
     Z3=Theta2*a2;
     a3=sigmoid(Z3);
	 y1=y(i,:);
     y1=y1';
     
	 A=log(a3);
     B=log(1-a3);
     D=(1-y1) .* B;
	 C=y1 .* A;
     E= - C - D;
     F=sum(E);
     G=F/m;
     J=J+G;     
end	 
s=0;
s1=0;
for i=1:hidden_layer_size,
    for j=2:(input_layer_size+1),
	s=s+Theta1(i,j)*Theta1(i,j);
	end
end
for i=1:num_labels,
    for j=2:(hidden_layer_size+1),
	  s1=s1+Theta2(i,j)*Theta2(i,j);
	end
end

s=s+s1;
s=s*lambda/(2*m);
J=J+s;

 

% Implementing backpropagation using Vectorization
a1=X;
Z2=X*Theta1';
a2=sigmoid(Z2);
a2=[ones(m,1) a2];
Z3=a2*Theta2';
a3=sigmoid(Z3);
y_matrix=y;
d3=a3-y_matrix;
d2=d3*Theta2(:,2:end) .* sigmoidGradient(Z2);
Theta1_grad=d2'*a1;
Theta2_grad=d3'*a2;

Theta1_grad=Theta1_grad/m;
Theta2_grad=Theta2_grad/m;


% Regularizing the gradient
m1=size(Theta1_grad,1);
n1=size(Theta1_grad,2);
m2=size(Theta2_grad,1);
n2=size(Theta2_grad,2);
   for i=1:m1,
       for j=2:n1,
	     Theta1_grad(i,j)=Theta1_grad(i,j)+(lambda*Theta1(i,j))/m; % all have already been divided by m. we have to add lambda*theta1(i,j)where j!=1
       end	 
   end 
   for i=1:m2,
      for j=2:n2,
	    Theta2_grad(i,j)=Theta2_grad(i,j)+(lambda*Theta2(i,j))/m;
      end
   end

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
