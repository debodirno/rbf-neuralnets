function [Centers, betas] = refineRBFN2(X_train, Y_train, X_activ, Centers, betas, Theta)
    % This code modifies (refines) the weights between the hidden layer and
    % the output layer.
    %
    % It takes in the following parameters :
    % 
    % X_train : training attributes
    % Y_train : corresponding classes of the samples
    % X_activ : activation outputs of hidden network
    % Centers : The prototype vectors stored in the RBF neurons.
    % betas   : The beta coefficient for each coressponding RBF neuron.(spread)
    % Theta   : The weights for the output layer. There is one row per neuron
    %           and one column per output node / category.
    %
    % It modifies the following parameters :
    % 
    % Centers
    % betas
    % 
    % The module runs iteratively over all the attributes and all training
    % samples and all the centers and all classes and checks the difference
    % between correct output and the obtained output and calculates the 
    % change using a learning parameter.
    %
    % It then subtracts this from all the elements of the matrix Theta,
    % which is thus, modified.
    
    numCats = size(unique(Y_train), 1);                             % number of categories of Y (number of classes)
    
    m = size(X_train, 1);                                           % number of rows in X_train
    
    numRBFNeurons = size(Centers, 1);                               % number of RBF Neurons in the hidden layer
    
    eta = 0.1;                                                      % learning rate
    del_center = zeros(size(Centers));                              % change in center
    del_betas = zeros(size(betas));                                 % change in betas
    
    for i = 1:size(Centers, 1)
        for mu = 1:m                                                % iterate over all training samples
            for j = 1:numRBFNeurons                                 % iterate over all centers
                for k = 1:numCats
                    scores = evaluateRBFN(Centers, betas, Theta, X_train(mu, :));   % returns a matrix
                    [maxScore, category] = max(scores);                             % calculate the largest in each column -> maxScore , 
                                                                                    % and also the indices of the rows where max occurs -> category
                    diff = 0;
                    del_center_i = 0;
                    del_betas_i = 0;
                    
                    for p = 1:numCats
                        diff = diff + Theta(j,p) * (category - Y_train(mu));   % calculate the difference between the maximum response for the sample and the original category of the mu-th sample
                    end

                    del_center_i = del_center_i + (((X_activ(mu ,j) * (X_train(mu, :) - Centers(j, :))) ./ (betas(j, :).^2)) * diff);  % calculate the change in center for a particular set of (mu, i, j, k) 
                    del_betas_i = del_betas_i + (((X_activ(mu, j) * (norm(X_train(mu, :) - Centers(j, :))^ 2)) ./ (betas(j, :).^3)) * diff);
                    
                    del_center(i, :) = del_center_i;
                    del_betas(i,:) = del_betas_i;
                end
            end
        end
    end
    
    del_center = eta * del_center;                                    % update del_center
    del_betas = eta * del_betas;                                      % update del_betas  
	
    fprintf('Del Center : %s\n', del_center);
    fprintf('Del Betas : %s\n', del_betas);
    
    Centers = Centers - del_center; 		% update Centers
	betas = betas - del_betas;              % update Betas