function [rmse, mae] = EvalPred(U, V, B, vec, a, b)
% a, b for scaling of the predicted result U*V*B'

% ---
UB = U*B;
Uvec = UB(vec(:,1), :);
Vvec = V(vec(:,2),:);
r = sum(Uvec .* Vvec,2); % Y.*(R-U*B*V')
if nargin == 6
    r=r*a+b;
end

% --- 
r(r>5)=5;
r(r<1)=1;

% --- MAE, RMSE
Diff = ( vec(:,3) - r ); % difference
rmse = sqrt( mean(Diff.^2) );
mae = mean( abs(Diff) );
