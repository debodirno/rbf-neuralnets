% load the som outputs in h , it will be a 3*4 matrix
% let x=input vector
clear;
data=csvread('iris.csv');
x=data(:,(1:4));
%x=zscore(x); % use only during data normalization
y=data(:,(5:7));
m=size(x,1);
%n=size(x,2);

load som_w.mat;
h=som_w; % the weights have been saved by the variable name som_w

input_layer_size  = 14;  % number of attributes
%hidden_layer_size = 25;   % 25 hidden units
num_labels = 3;   % no of classes         
initial_Theta1 = randInitializeWeights(input_layer_size , num_labels);


hrow=size(h,1);

%computed the distance of n-1 SOM nodes from the first node
hdist=[0];
for i=2:size(h,1)
 d=dist(h(1,:),h(i,:)');
 hdist=[hdist;d];
end

htemp=[];
for i=1:size(h,1)
 htemp=[htemp ; h(i,:) , hdist(i,1)];
end
%fu=htemp;
htemp=sortrows(htemp,size(htemp,2));
fu2=h;
h=htemp(:,1:(size(htemp,2)-1)); % removing the last column which has the distance of each neuron from the first neuron

sigma=zeros(hrow,1);
for i=1:hrow,
    
    if i==1
      sigma(i,1)=dist(h(1,:),h(2,:)')/2;
    else if i==size(h,1)
      sigma(i,1)=dist(h(i,:),h(i-1,:)')/2;
        else
        d=[];
        d=dist(h(i,:),h(i-1,:)');
        d=dist(h(i,:),h(i+1,:)');
        d=dist(h(i-1,:),h(i+1,:)');
        minimum=min(d);
        sigma(i,1)=minimum/3;
        end
    end
            
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
alpha=0.1;
num_iters = 10000;
X=a1;
[theta, J_history] = gradientDescent(X, y, initial_Theta1, alpha, num_iters,input_layer_size, num_labels);


Theta1=theta;
pred = predict(Theta1,X);
sum=0;
for i=1:size(X,1)
for j=1:num_labels
 if y(i,j)==1 && j==pred(i)
     sum=sum+1;
 end
end
end
fprintf('\nTraining Set Accuracy: %f\n', sum/m * 100);


 
 

