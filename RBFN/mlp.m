clc;
clear all;
accuracy = 0;

% habermantest;
% balancetest
% brcancer
% glass
% wine;
% Iris;
% IONOS;
 liver;
% sonar;
% thyroid;
% vehicle;
% seeds;
% tae;
% pima
% WDBC

noh = 10;
for epoch = 1:1
accuracy_epoch = 0;
for j = 1:10
    count = 0;
	indices = crossvalind('Kfold',T1,10);
    test = (indices == j);
    train1 = ~test;
    
    batchdata = batchdata1(train1,:);
	[batchdata, MU, SIGMA] = zscore(batchdata);
	P = batchdata';
    T = Te(train1,:)';
      
    net = feedforwardnet(noh,'traingd');
    net.trainParam.epochs = 15000;
    net.trainParam.lr = 0.1;
    net.trainParam.showWindow = false;

    [net] = train(net,P,T);
    
    testbatchdata = batchdata1(test,:);
%	testbatchdata = zscore(testbatchdata);
    testbatchdata = (testbatchdata-repmat(MU,size(testbatchdata,1),1))./repmat(SIGMA,size(testbatchdata,1),1);   % zscore normalization of test data
	X_tar = testbatchdata';
    Test = Te(test,:)';
    Test = vec2ind(Test); 
    
    Y = net(X_tar);
    Yc = vec2ind(Y);  
     
    for i = 1:size(Test,2)
        if Yc(i)~= Test(i)
            count = count+1;
        end
    end
    accuracy_epoch = accuracy_epoch+(size(X_tar,2)-count) /size(X_tar,2)*100   
end
    accuracy_epoch = accuracy_epoch/10;
    sd(epoch) = accuracy_epoch;
    accuracy = accuracy + accuracy_epoch;
end
accuracy = accuracy/1
standard_daviation = std(sd)

csvwrite([data num2str(noh) '.txt'], [accuracy standard_daviation] );

