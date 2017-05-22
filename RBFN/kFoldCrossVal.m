clc;
clear all;
accuracy = 0;

addpath('kMeans');
addpath('RBFN');

filename  = input('Enter the input file name ','s');
%attr_beg  = input('Enter the beginning column of attributes ');
attr_end  = input('Enter the ending column of attributes ');
%class_col = input('Enter the column of target class ');
class_col = attr_end + 1;
number_of_class = input('Enter the number of classes ');
%number_of_attr  = input('Enter the number of attributes ');
number_of_attr = attr_end;
training_type = input('Enter the hidden node selection type 1.K-means , 2. Random Selection , 3. SOM 4. Noise Induced SOM ');
noise = input('Enter percentage of noise ');
if (noise == 0)
    type_of_noise = 1;
else
    type_of_noise  = input('Enter the choice 1.Additive 2. Multiplicative ');       %type of noise
end
centersPerCategory = number_of_attr;
%batchdata1 = load('iris.csv');
batchdata1 = load(filename);
%batchdata1(1,1)
conf = zeros(number_of_class,number_of_class);          % confidence matrix, here 3 x 3 matrix
% Set 'm' to the nudatamber of data points.
m = size(batchdata1, 1);    % returns the number of rows in batchdata1, here 150
%-----------------------------------------------------------------------------------------
% noise addition
%-----------------------------------------------------------------------------------------
for i = 1:m     %
    for j = 1:attr_end
        if type_of_noise == 1   % choice over additive and multiplicative
            %Additive noise
            batchdata1(i, j) = batchdata1(i, j)*(1 + noise);
        else
            %Multiplicative noise
            batchdata1(i, j) = batchdata1(i, j)*(1 + (batchdata1(i, j)* noise)/100);
        end
    end
end
%-----------------------------------------------------------------------------------------
%
%-----------------------------------------------------------------------------------------
accuracy_epoch2 = 0;

for epoch = 1:10 % make this 5 number of times
  
  accuracy_epoch = 0;
  conf_epoch = zeros(number_of_class,number_of_class);
  
  for j = 1:10   %number of folds

    count = 0;
	
    %indices = crossvalind('Kfold',T1,10); % T1=150
    indices = crossvalind('Kfold',m,10); % T1=150           % generate random indices for a 10 fold cross validation of 150 observations
    %indices = crossvalind('Kfold',178,10);
    
    test = (indices == j);
    train1 = ~test;
    
    batchdata = batchdata1(train1,:); %%%%%%%%% collects the elements from K-1 folds except the Kth fold
    
    X = batchdata(:,1 :attr_end);  % X stores the all the data from all rows and from column 1 through attr_end
    Y = batchdata(:,class_col );   %class_col is next to attr_end
    [X, MU, SIGMA] = zscore(X);    % X -> Z score, MU -> mean, SIGMA -> standard deviation  % X = (x - MU) / SIGMA 
        
    [Centers, betas, Theta, X_activ] = trainRBFN(X, Y , centersPerCategory , true, training_type);       % train the RBF Network
    
    %[Theta] = refineRBFN(X, Y, X_activ, Centers, betas, Theta);
    %[Centers, betas] = refineRBFN2(X, Y, X_activ, Centers, betas, Theta);
    batchdata_test = batchdata1(test,:);        % select all items in the test row of batchdata1
    X_test = batchdata_test(:,1 :attr_end);     % select all items in the (1 through attr_end) columns of batchdata_test
    Y_test = batchdata_test(:, class_col);      % select all items in the class_col column of batchdata_test
    
    X_test = (X_test-repmat(MU,size(X_test,1),1))./repmat(SIGMA,size(X_test,1),1);   % zscore normalization of test data
    X_test = X_test + random('norm',0,noise,size(X_test));
    
    [accuracy,confusion]=testing(X_test, Y_test, Centers, betas, Theta, number_of_class);  
    
    accuracy_epoch = accuracy_epoch + accuracy;         % calculate accuracy for the epoch
    conf_epoch = conf_epoch + confusion;                % calculate confusion for the epoch
    
  end
  
  accuracy_epoch2 = accuracy_epoch2 + accuracy_epoch / 10;      
  sd(epoch) = accuracy_epoch / 10;                      % create a vector for the epoch-th testing accuracy to calculate the SD
  conf = conf + conf_epoch / 10;
end

accuracy_epoch2 = accuracy_epoch2 / 10;  % accuracy_epoch2 calculates the average accuracy for the data set
standard_deviation = std(sd);            % standard deviation for the data set
conf = conf/10;                          % conf is a average confusion matrix for the data set


fprintf('accuracy : %f\n',  accuracy_epoch2 );
fprintf('standard deviation : %f\n',  standard_deviation );

dlmwrite(['output_' filename num2str(training_type) '_' num2str(noise) '_' datestr(now, 'yyyymmddHHMMSS') '.txt'], conf);
dlmwrite(['output_' filename num2str(training_type) '_' num2str(noise) '_' datestr(now, 'yyyymmddHHMMSS') '.txt'], ' ', '-append');
dlmwrite(['output_' filename num2str(training_type) '_' num2str(noise) '_' datestr(now, 'yyyymmddHHMMSS') '.txt'], [accuracy_epoch2 standard_deviation], '-append');


