
clear;

% --- load data
load CSTdata.mat;
    % train_vec: (user, item, raring)
    % probe_vec: (user, item, raring)
    % test_vec:  (user, item, raring)
    % U0, V0: the coordinate systems can be estimated using svds or CST   

% --- parameters
tradeoff_lambda = 1; % tradeoff parameter

% --- training via CST
profile on
[U, B, V, RMSE_tr, RMSE_pr, MAE_tr, MAE_pr] = CST(train_vec, probe_vec, U0, V0, tradeoff_lambda);
profile off

% --- prediciton on the test data
[rmse, mae] = EvalPred(U,V,B,test_vec,1,0);
