function [Theta] = refineRBFN(X_train, Y_train, X_activ, Centers, betas, Theta)
    
    numCats = size(unique(Y_train), 1);         % number of categories of Y (number of classes)
    
    m = size(X_train, 1);                       % number of rows in X_train
    
    numRBFNeurons = size(Centers, 1);           % number of RBF Neurons in the hidden layer
    
    eta = 0.01;                                 % learning rate
    del_theta = 0;                              % change in theta

    for mu = 1:m                                % iterate over all training samples
        for j = 1:numRBFNeurons                 % iterate over all centers
            for k = 1:numCats                   % iterate over all classes

                scores = evaluateRBFN(Centers, betas, Theta, X_train(mu, :));     % returns a matrix

                [maxScore, category] = max(scores);             % calculate the largest in each column -> maxScore , 
                                                                % and also the indices of the rows where max occurs -> category
                
                diff = category - Y_train(mu);                  % calculate the difference between the maximum response for the sample and the original category of the mu-th sample
                del_theta = del_theta + (X_activ(mu ,j) * diff); % calculate the change in theta for a particular set of (mu, j, k) 
            end
        end
    end
    
    del_theta = eta * del_theta;        % update del_theta
    
    Theta = Theta - del_theta;          % update Theta