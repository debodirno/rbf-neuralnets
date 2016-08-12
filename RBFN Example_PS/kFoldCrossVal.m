clc;
clear all;
accuracy = 0;

addpath('kMeans');
addpath('RBFN');

filename  = input('Enter the input file name','s');
attr_beg  = input('Enter the beginning column of attributes');
attr_end  = input('Enter the ending column of attributes');
class_col = input('Enter the column of target class');
number_of_class = input('Enter the nummber of classes');
number_of_attr  = input('Enter the number of attributes');
training_type = input('Enter the hidden node selection type 1.K-means , 2. Random Selection , 3. SOM');
centersPerCategory = input('Enter the number of centers per category');
%batchdata1 = load('iris.csv');
batchdata1 = load(filename);

% Set 'm' to the number of data points.
m = size(batchdata1, 1);

accuracy_epoch2=0;
for epoch = 1:5 % make this 5
  accuracy_epoch = 0;
  for j = 1:10
    count = 0;
	%indices = crossvalind('Kfold',T1,10); % T1=150
    indices = crossvalind('Kfold',m,10); % T1=150
    %indices = crossvalind('Kfold',178,10);
    test = (indices == j);
    train1 = ~test;
    
    batchdata = batchdata1(train1,:); %%%%%%%%% collects the elements from K-1 folds except the Kth fold
    
    X = batchdata(:,attr_beg :attr_end);
    Y = batchdata(:,class_col );
    
    [Centers, betas, Theta] = trainRBFN(X, Y , centersPerCategory , true,training_type);
    
    batchdata_test = batchdata1(test,:);
    X_test = batchdata_test(:,attr_beg :attr_end);
    Y_test = batchdata_test(:, class_col);
    accuracy=testing(X_test, Y_test, Centers, betas, Theta);
    accuracy_epoch=accuracy_epoch+accuracy;
end
accuracy_epoch2=accuracy_epoch2+accuracy_epoch/10;
end
accuracy_epoch2=accuracy_epoch2/5;
fprintf('%f',  accuracy_epoch2 );

