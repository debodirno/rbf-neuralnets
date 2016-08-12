clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 4;  % 4 attributes
hidden_layer_size = 25;   % 25 hidden units
num_labels = 3;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading Data =============
%  We start the exercise by first loading and visualizing the dataset. 
% Load Training Data
fprintf('Loading Data ...\n');
data=csvread('iris.csv');
x=data(:,(1:4));
y=data(:,(5:7));

m=size(x,1);

%% ================ Part 2: Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =================== Part 3: Training NN ===================

fprintf('\nTraining Neural Network... \n')
lambda=1;
options = optimset('MaxIter', 500);
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, x, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');



%% ================= Part 4: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, x);

sum=0;
for i=1:size(x,1)
for j=1:num_labels
 if y(i,j)==1 && j==pred(i)
     sum=sum+1;
 end
end
end
fprintf('\nTraining Set Accuracy: %f\n', sum/m * 100);


