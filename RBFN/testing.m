function [accuracy,confusion]=testing(X, y, Centeres, betas, Theta,number_of_class)
% ========================================
%       Measure Training Accuracy
% ========================================

disp('Measuring training accuracy...');

numRight = 0;
op = zeros(1,number_of_class)+0.00000001;
confusion = zeros(number_of_class,number_of_class); 
wrong = [];
category = [];
m=size(X,1);

% For each training sample...
for (i = 1 : m)
    % Compute the scores for both categories.
    scores = evaluateRBFN(Centeres, betas, Theta, X(i, :));     % returns a matrix
    
	[maxScore, category] = max(scores);             % calculate the largest in each column -> maxScore , 
                                                    % and also the indices of the rows where max occurs -> category
	
    % Validate the result.
    if (category == y(i))
        numRight = numRight + 1;
    else
        wrong = [wrong; X(i, :)];
    end
    
    op(y(i)) = op(y(i))+1;
    confusion(category,y(i)) = confusion(category,y(i))+1;  
    
end

for i=1:size(unique(y),1)
            confusion(:,i) = confusion(:,i)/op(i)*100; 
end

% Mark the incorrectly recognized samples with a black asterisk.
%plot(wrong(:, 1), wrong(:, 2), 'k*');

accuracy = numRight / m * 100;
fprintf('Training accuracy: %d / %d, %.1f%%\n', numRight, m, accuracy);
if exist('OCTAVE_VERSION') fflush(stdout); end;

