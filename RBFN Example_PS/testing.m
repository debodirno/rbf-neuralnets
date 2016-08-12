function [accuracy]=testing(X, y, Centeres, betas, Theta)
% ========================================
%       Measure Training Accuracy
% ========================================

disp('Measuring training accuracy...');

numRight = 0;

wrong = [];
m=size(X,1);
% For each training sample...
for (i = 1 : m)
    % Compute the scores for both categories.
    scores = evaluateRBFN(Centeres, betas, Theta, X(i, :));
    
	[maxScore, category] = max(scores);
	
    % Validate the result.
    if (category == y(i))
        numRight = numRight + 1;
    else
        wrong = [wrong; X(i, :)];
    end
    
end

% Mark the incorrectly recognized samples with a black asterisk.
%plot(wrong(:, 1), wrong(:, 2), 'k*');

accuracy = numRight / m * 100;
fprintf('Training accuracy: %d / %d, %.1f%%\n', numRight, m, accuracy);
if exist('OCTAVE_VERSION') fflush(stdout); end;
