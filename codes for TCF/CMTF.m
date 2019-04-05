function [U_, B_, V_, B_aux_, RMSE_tr, RMSE_te, MAE_tr, MAE_te] = CMTF( train_vec, aux_vec, probe_vec, para, test_vec )
% target: {1,2,3,4,5,?}, and we use {1,2,3,4,5,0} in MATLAB 
    % (has been scaled to [0,1]) already
% auxilary: {0,1,?}, and we use {eps,1,0} in MATLAB

% --- parameters
tradeoff_lambda     = para.tradeoff_lambda;
tradeoff_alpha_U    = para.tradeoff_alpha_U;
tradeoff_alpha_V    = para.tradeoff_alpha_V;
tradeoff_beta     = para.tradeoff_beta;
tradeoff_beta_aux = para.tradeoff_beta_aux;

num_feat = para.num_feat;
num_user = para.num_user;
num_item = para.num_item;

MAX_EPOCH = para.MAX_EPOCH;

%% data structures to save the result
RMSE_tr  = zeros(MAX_EPOCH, 1);
MAE_tr   = zeros(MAX_EPOCH, 1);
RMSE_te  = zeros(MAX_EPOCH, 1); 
MAE_te   = zeros(MAX_EPOCH, 1);
RMSE_pr  = zeros(MAX_EPOCH, 1); 
MAE_pr   = zeros(MAX_EPOCH, 1);

%% epoch 1
i = 1;

% --- initializations
rand('state',0);
U = mean(train_vec(:,3))/num_feat + 0.01 * (rand(num_user, num_feat) - 0.5);
rand('state',1); 
V = mean(train_vec(:,3))/num_feat + 0.01 * (rand(num_item, num_feat) - 0.5);

% --- Estimate B, B_aux
B = EstimateB(U,V,train_vec,tradeoff_beta);
B_aux = EstimateB(U,V,aux_vec,tradeoff_beta_aux);

% --- save the parameters
U_ = U; V_ = V; B_ = B; B_aux_ = B_aux;

[rmse, mae] = EvalPred(U,V,B,[train_vec(:,1:2), train_vec(:,3)*4+1],4,1); RMSE_tr(i) = rmse; MAE_tr(i) = mae;
[rmse, mae] = EvalPred(U,V,B,[test_vec(:,1:2),  test_vec(:,3)*4+1], 4,1); RMSE_te(i) = rmse; MAE_te(i) = mae;
[rmse, mae] = EvalPred(U,V,B,[probe_vec(:,1:2),  probe_vec(:,3)*4+1], 4,1); RMSE_pr(i) = rmse; MAE_pr(i) = mae;

fprintf(1, 'epoch: %d, tr: %6.4f(RMSE), %6.4f(MAE); te: %6.4f(RMSE), %6.4f(MAE); pr: %6.4f(RMSE), %6.4f(MAE) \n', i, RMSE_tr(i), MAE_tr(i),  RMSE_te(i), MAE_te(i), RMSE_pr(i), MAE_pr(i));

%% epoch 2 ...
vec_V = [train_vec; [aux_vec(:,1)+num_user, aux_vec(:,2:3)] ]; % for update item-specific latent features
vec_U = [train_vec; [aux_vec(:,1), aux_vec(:,2)+num_item, aux_vec(:,3)] ]; % for update user-specific latent features

for i = 2 : MAX_EPOCH 
 
    % --- update V
    U2 = [U*B; U*B_aux];
    V = MF_U(vec_V(:,[2 1 3]), tradeoff_alpha_V, V, U2, tradeoff_lambda, num_user);
    
    % --- update U
    V2 = [V*B'; V*B_aux'];
    U = MF_U(vec_U, tradeoff_alpha_U, U, V2, tradeoff_lambda, num_item);
    % ----------------------------------------         

    % ----------------------------------------           
    % --- Estimate B, B_aux
    B = EstimateB(U,V,train_vec,tradeoff_beta);
    B_aux = EstimateB(U,V,aux_vec,tradeoff_beta_aux);

    % ----------------------------------------------------------------
    [rmse, mae] = EvalPred(U,V,B,[train_vec(:,1:2), train_vec(:,3)*4+1],4,1); RMSE_tr(i) = rmse; MAE_tr(i) = mae;
    [rmse, mae] = EvalPred(U,V,B,[test_vec(:,1:2), test_vec(:,3)*4+1],4,1);  RMSE_te(i) = rmse; MAE_te(i) = mae;
    [rmse, mae] = EvalPred(U,V,B,[probe_vec(:,1:2),  probe_vec(:,3)*4+1], 4,1); RMSE_pr(i) = rmse; MAE_pr(i) = mae;
    fprintf( 1, 'epoch: %d, tr: %6.4f(RMSE), %6.4f(MAE); te: %6.4f(RMSE), %6.4f(MAE); pr: %6.4f(RMSE), %6.4f(MAE) \n ', i, RMSE_tr(i), MAE_tr(i),  RMSE_te(i), MAE_te(i),  RMSE_pr(i), MAE_pr(i)  );

 
    % ----------------------------------------------------------------
    % --- Check for convergence
    if i > 1
        if (RMSE_te(i-1) < RMSE_te(i)) || (abs( RMSE_te(i-1) - RMSE_te(i) ) <1E-4)
            break;
        end
    end
    % ----------------------------------------------------------------
    % --- save the parameters
    U_ = U; V_ = V; B_ = B; B_aux_ = B_aux;
    
end % end of epoch

%% Results
RMSE_tr  = RMSE_tr(RMSE_tr>0);
MAE_tr   = MAE_tr(MAE_tr>0);
RMSE_te   = RMSE_te(RMSE_te>0);
MAE_te    = MAE_te(MAE_te>0);
RMSE_pr = RMSE_pr(RMSE_pr>0);
MAE_pr  = MAE_pr(MAE_pr>0);


%% Update U in single Matrix
function [U] = MF_U(vec, tradeoff_alpha, U, V, tradeoff_lambda, num_item)
% vec (user, item, rating): num_rating * 3
  
% ---------------------------------------
num_user = size(U,1);

for u = 1 : num_user
    % --- (user, item, rating) of the current user
    vec_u = vec(vec(:,1) == u, :);
    
    if ~isempty(vec_u)
        % --- item-specific latent features
        V_u = V( vec_u(:,2), : );    % items rated by the current user: u

        f0 = (vec_u(:,2) <= num_item); % target items
        f1 = (vec_u(:,2) > num_item); % auxiliary items
        
        % Solve the Linear System
        b = vec_u(f0,3)' * V_u(f0,:) + tradeoff_lambda * vec_u(f1,3)' * V_u(f1,:);
        A = V_u(f0,:)'*V_u(f0,:) + tradeoff_lambda*V_u(f1,:)'*V_u(f1,:) + tradeoff_alpha*(sum(f0)+sum(f1)*tradeoff_lambda)*eye(size(V_u,2));
        
        % U(u,:) = b * inv(A);
        U(u,:) = b/A;
    end
end

