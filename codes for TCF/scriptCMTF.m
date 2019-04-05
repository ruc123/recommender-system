clear;

load TCFdata.mat;
% train_vec: numerical ratings
% aux_vec: {0,1} binary ratings, and we use eps for 0 in MATLAB
% test_vec: numerical ratings
% probe_vec: numerical ratings

%%
para.MAX_EPOCH = 100;
para.num_user = 21718;
para.num_item = 14301;

para.num_feat = 50;
para.tradeoff_lambda = 0.1;
para.tradeoff_beta = 1;
para.tradeoff_beta_aux = 1;
para.tradeoff_alpha_U = 0.1;
para.tradeoff_alpha_V = 0.1;

% --- Collective Matrix Tri-Factorization (CMTF)
% Scale from 1-5 to 0-1
train_vec(:,3) = ( train_vec(:,3)-1 )/4;
train_vec( train_vec(:,3)==0, 3 ) = eps; 

probe_vec(:,3) = ( probe_vec(:,3)-1 )/4;
probe_vec( probe_vec(:,3)==0, 3 ) = eps;

test_vec(:,3) = ( test_vec(:,3)-1 )/4;
test_vec( test_vec(:,3)==0, 3 ) = eps;





profile on
    [U, B, V, B_aux, RMSE, RMSE_te, MAE, MAE_te] = CMTF(train_vec, aux_vec, probe_vec, para, test_vec);
profile off

[rmse, mae] = EvalPred(U,V,B,test_vec,4,1)
