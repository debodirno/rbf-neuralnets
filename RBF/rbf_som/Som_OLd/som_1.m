data = csvread('iris.csv');
batchdata = data(:,1:4);
[dataRow, dataCol] = size(batchdata');
[dataRow, dataCol] = size(batchdata'); % sizo of batchdata sample_size*sample_attributes
%lbl = Te;
%[mVal mInd] = max(lbl');
%mInd = mInd';
somRow = 3;
som_w = zeros(somRow, dataRow);
euclideanD = zeros(somRow);
neighbourhoodF = zeros(somRow);
%count = zeros(somRow,size(lbl,2));
win = zeros(somRow);
iter = 5*somRow;
eta0 = 0.9;
etaN = eta0;
sig0 = 5;
sigN = sig0;
tau1 = iter/log(sig0);
tau2 = iter;
for r = 1:somRow
        som_w(r, :) = 2*rand(dataRow, 1)-1;
end
batchdata = batchdata';
for i = 1:iter
    variance = sigN^2;
    for j = 1:dataCol
    for r = 1:somRow
            v = batchdata(:,j) - som_w(r,:)';
            euclideanD(r) = sqrt(v' * v);
    end
    % winner neuron on the SOM Map
    [vect,winnerRow]=min(euclideanD); % 1 stands for 1st dimension, i.e. row
    %[winnerEuclidean, winnerCol]=min(vector,[],2); % 2 stands for 2nd dimension, i.e. column
    %winnerRow = winnerRowVector(winnerCol);
    
    for r = 1:somRow
            if (r == winnerRow(1))   % Is the winner
                neighbourhoodF(r) = 1;
                continue;
            else   % Not the winner
                distance = (winnerRow(1) - r);
                neighbourhoodF(r) = exp(-distance/(2*variance));
            end
    end
    
    for r = 1:somRow
            oldWeightVector = som_w(r,:)';
            % Update weight vector of neuron
            som_w(r,:) = oldWeightVector + etaN*neighbourhoodF(r)*(batchdata(:,j) - oldWeightVector);
    end
    end
    if (i > iter*0.9)
        etaN = 0.05;
        sigN = 1;
    else
        etaN = eta0*exp(-i/tau2);
        sigN = sig0*exp(-i/tau1);
        eta = etaN;
        sig = sigN;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
output = [];


for j = 1:dataCol
    for r = 1:somRow
            v = batchdata(:,j) - som_w(r,:)';
            euclideanD(r) = sqrt(v' * v);
    end
    [vector, winnerRow] = min(euclideanD); % 1 stands for 1st dimension, i.e. row
    %[winnerEuclidean, winnerCol] = min(vector,[],2); % 2 stands for 2nd dimension, i.e. column
    %winnerRow = winnerRowVector(winnerCol);
    output       = [output;som_w(winnerRow(1))];
end
%}
    %%%%%%%output is the SOM map output in place of input samples.