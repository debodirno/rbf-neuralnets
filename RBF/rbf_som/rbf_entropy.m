% load the som outputs in h , it will be a 3*4 matrix
% let x=input vector
data=csvread('iris.csv');
x=data(:,(1:4));
x=zscore(x); % use only during data normalization
y=data(:,(5:7));
m=size(x,1);
%n=size(x,2);

load som_w.mat;
h=som_w; % the weights have been saved by the variable name som_w

input_layer_size  = 14;  % number of attributes
%hidden_layer_size = 25;   % 25 hidden units
num_labels = 3;   % no of classes         


hrow=size(h,1);

%compute sigma p has been taken as two for iris
sigma=zeros(hrow,1);
for i=1:hrow,
    sum=0;
    for j=1:hrow
      if j~=i
        sq=(dist(h(i,:),h(j,:)')).^2;
        sum=sum+sq;
      end
    end
    sum=sum/2;
    sigma(i,1)=sqrt(sum);
end


% we will calculate the activation for hidden layer

a1=zeros(m,size(h,1)); % a2 will be 150*3


for i=1:m
  for j=1:hrow
    a1(i,j)=exp(( - ((dist(h(j,:),x(i,:)')) .^2))/ (2*((sigma(j,1)).^2)));
  end
end

% now treat the hidden layer and the output layer as a SLP
%the inputs of the SLP is in a2 
%add a bias node to a2

X=a1;
 
initial_Theta1 = randInitializeWeights(input_layer_size , num_labels);

 
 
initial_nn_params = [initial_Theta1(:)]; 
options = optimset('MaxIter', 500);

%  You should also try different values of lambda
%lambda = 1;

% Create "short hand" for the cost function to be minimized
%costFunction = @(p) nnCostFunction(p, ...
%                                   input_layer_size, ...
%                                   hidden_layer_size, ...
%                                   num_labels, X, y, lambda);
                               
costFunction = @(p)CostFunc(p, ...
                                   input_layer_size, ...
                                    num_labels, X, y);
                               


% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
% Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                  hidden_layer_size, (input_layer_size + 1));

Theta1 = reshape(nn_params(1:num_labels * (input_layer_size + 1)), ...
                 num_labels, (input_layer_size + 1));
             
pred = predict(Theta1,X);

%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
sum=0;
for i=1:size(X,1)
for j=1:num_labels
 if y(i,j)==1 && j==pred(i)
     sum=sum+1;
 end
end
end
fprintf('\nTraining Set Accuracy: %f\n', sum/m * 100);


 
 

