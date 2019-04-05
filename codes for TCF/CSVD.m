function [U_ B_ V_ B_aux_ RMSE_tr RMSE_te MAE_tr MAE_te] = CSVD(train_vec, aux_vec, test_vec, para, probe_vec)
% target: {1,2,3,4,5,?}, and we use {1,2,3,4,5,0} in MATLAB 
    % (has been scaled to [0,1]) already
% auxilary: {0,1,?}, and we use {eps,1,0} in MATLAB

%% parameters
MAX_EPOCH           = para.MAX_EPOCH;
num_user            = para.num_user;
num_item            = para.num_item;
num_feat            = para.num_feat;
tradeoff_lambda     = para.tradeoff_lambda;
tradeoff_beta       = para.tradeoff_beta;
tradeoff_beta_aux   = para.tradeoff_beta_aux;

%%
R = spconvertFULL(train_vec,num_user,num_item);
R_aux = spconvertFULL(aux_vec,num_user,num_item);

%% vector for the result
RMSE_tr = zeros(MAX_EPOCH, 1);
RMSE_te = zeros(MAX_EPOCH, 1);
MAE_tr = zeros(MAX_EPOCH, 1);
MAE_te = zeros(MAX_EPOCH, 1);
RMSE_pr = zeros(MAX_EPOCH, 1);
MAE_pr = zeros(MAX_EPOCH, 1);


%% epoch 1
% --- initialization
if para.init_aux == true
    % - initialization using the auxiliary data
    [U B V] = svds(R_aux,num_feat);
else
    % - initialization using the target data
    [U B V] = svds(R,num_feat);
end

% ------------------------------------------------------------------
% --- Estimate B, B_aux
B       = EstimateB(U,V,train_vec,tradeoff_beta);
B_aux   = EstimateB(U,V,aux_vec,tradeoff_beta_aux);
% ------------------------------------------------------------------

% --- save the parameters
U_ = U; V_ = V; B_ = B; B_aux_ = B_aux;

% --- Check Performance
i=1;
[rmse, mae] = EvalPred(U,V,B,[train_vec(:,1:2), train_vec(:,3)*4+1],4,1); RMSE_tr(i) = rmse; MAE_tr(i) = mae;
[rmse, mae] = EvalPred(U,V,B,[test_vec(:,1:2),  test_vec(:,3)*4+1], 4,1); RMSE_te(i) = rmse; MAE_te(i) = mae;
[rmse, mae] = EvalPred(U,V,B,[probe_vec(:,1:2),  probe_vec(:,3)*4+1], 4,1); RMSE_pr(i) = rmse; MAE_pr(i) = mae;

fprintf( 1, 'epoch: %d, tr: %6.4f(RMSE), %6.4f(MAE); te: %6.4f(RMSE), %6.4f(MAE); pr: %6.4f(RMSE), %6.4f(MAE) \n ', i, RMSE_tr(i), MAE_tr(i),  RMSE_te(i), MAE_te(i),  RMSE_pr(i), MAE_pr(i)  );

%% epch 2 ...
for i = 2 : MAX_EPOCH

% step 1 and 2 for estimating U and V, respectively
    % -------------------------------------------------------------
    for iterUV = 1:20
        rmse0=rmse; % the previous RMSE result
        
        % step 1: update U
            GradU = CalGradU(U,V,B,B_aux,train_vec,aux_vec,tradeoff_lambda);
            gamma = StepSize_gamma( U, GradU, V, B, B_aux, train_vec, aux_vec, tradeoff_lambda );
            U = U - gamma*GradU;

        % step 2: update V    
            GradV = CalGradU(V,U,B',B_aux',train_vec(:,[2 1 3]),aux_vec(:,[2 1 3]),tradeoff_lambda);
            gamma = StepSize_gamma( V, GradV, U, B', B_aux', train_vec(:,[2 1 3]), aux_vec(:,[2 1 3]), tradeoff_lambda );
            V = V - gamma*GradV;

        [rmse, mae] = EvalPred(U,V,B,test_vec,1,0);

        % --- stopping criteria for the inner loop of updating U and V
        if i > 1
            if ( abs( rmse0-rmse ) < 1E-4 ) || ( rmse0 < rmse )
                break;
            end
        end
        % ---    
    end
    % -------------------------------------------------------------
    fprintf( 1, 'iterUV: %d \n', iterUV);

% step 3: update B    
    % ------------------------------------------------------------------
    % * closed form solution
    B       = EstimateB(U, V, train_vec, tradeoff_beta);
    B_aux   = EstimateB(U, V, aux_vec, tradeoff_beta_aux);
    % ------------------------------------------------------------------

% --- Check Performance    
    [rmse, mae] = EvalPred(U,V,B,[train_vec(:,1:2), train_vec(:,3)*4+1],4,1); RMSE_tr(i) = rmse; MAE_tr(i) = mae;
    [rmse, mae] = EvalPred(U,V,B,[test_vec(:,1:2),  test_vec(:,3)*4+1], 4,1); RMSE_te(i) = rmse; MAE_te(i) = mae;
    [rmse, mae] = EvalPred(U,V,B,[probe_vec(:,1:2),  probe_vec(:,3)*4+1], 4,1); RMSE_pr(i) = rmse; MAE_pr(i) = mae;

    
    fprintf(1, 'epoch: %d, tr: %6.4f(RMSE), %6.4f(MAE); te: %6.4f(RMSE), %6.4f(MAE); pr: %6.4f(RMSE), %6.4f(MAE) \n', i, RMSE_tr(i), MAE_tr(i),  RMSE_te(i), MAE_te(i),  RMSE_pr(i), MAE_pr(i) );
    % ---

% --- stopping criteria
    if i > 1
        if ( abs( RMSE_te(i-1) - RMSE_te(i) ) < 1E-4 ) || ( RMSE_te(i-1) < RMSE_te(i) )
            break;
        end
    end
    % ---
    
    % --- save the parameters
    U_ = U; V_ = V; B_ = B; B_aux_ = B_aux;    
end

%% Result
RMSE_tr = RMSE_tr( RMSE_tr>0 );
RMSE_te = RMSE_te( RMSE_te>0 );
MAE_tr  = MAE_tr( MAE_tr>0 );
MAE_te  = MAE_te( MAE_te>0 );
RMSE_pr = RMSE_pr(RMSE_pr>0);
MAE_pr  = MAE_pr(MAE_pr>0);


%% calculate the gradient
function GradU = CalGradU(U, V, B, B_aux, train_vec, aux_vec, tradeoff_lambda)

% --- number of users and items
num_user = size(U,1);
num_item = size(V,1);

% * * * * * * * * * * * * target CF task * * * * * * * * * * * *
% --- 
uIDX    = train_vec(:,1);
vIDX    = train_vec(:,2);

% ---
VB  = V*B';

% ---
U2      = U(uIDX,:);
VB2     = VB(vIDX,:);

% ---
r       = sum(U2.*VB2,2);  % U*B*V'
Diff    = train_vec(:,3)-r;   % Y .* (R - U*B*V')
P       = spconvertFULL([uIDX, vIDX, Diff], num_user, num_item);  % time !!
Pu       = P * VB;

% * * * * * * * * * * * * auxiliary CF task * * * * * * * * * * * *
% --- 
uIDX = aux_vec(:,1);
vIDX = aux_vec(:,2);

% ---
VB  = V*B_aux';

% ---
U2      = U(uIDX,:);
VB2     = VB(vIDX,:);

% ---
r       = sum(U2.*VB2,2);  % U*B*V'
Diff    = aux_vec(:,3)-r;   % Y .* (R - U*B*V')
P       = spconvertFULL([uIDX, vIDX, Diff], num_user, num_item);  % time !!
Pu_aux  = P * VB;

% * * * * * * * * * * * * * * * * * * * * * * * *
Gu      = - Pu - tradeoff_lambda * Pu_aux;

% --- obtain Qu to project the gradient to the tangent space at U
Qu = - U'* Gu;

% --- final gradient
GradU = Gu + U*Qu;



%% line search
function gamma = StepSize_gamma( U, GradU, V, B, B_aux, train_vec, aux_vec, tradeoff_lambda )

% * * * * * * * * * * * * target CF task * * * * * * * * * * * *
% --- user-index, item-index
uIDX    = train_vec(:,1); 
vIDX    = train_vec(:,2);

% ---
VB      = V*B';   % V*B'

% ---
VB2    = VB(vIDX,:);

% --- 
U2      = U(uIDX,:);
r       = sum(U2.*VB2,2); % U*B*V'
x       = train_vec(:,3) - r; 

% ---
Gvec    = GradU(uIDX,:);
y      = sum(Gvec.*VB2,2); % GradU*B*V'

% * * * * * * * * * * * * auxiliary CF task * * * * * * * * * * * *
% --- user-index, item-index
uIDX    = aux_vec(:,1); 
vIDX    = aux_vec(:,2);

% ---
VB      = V*B_aux';   % V*B'

% ---
VB2    = VB(vIDX,:);

% --- 
U2      = U(uIDX,:);
r       = sum(U2.*VB2,2); % U*B*V'
x_aux       = aux_vec(:,3) - r; 

% ---
Gvec    = GradU(uIDX,:);
y_aux   = sum(Gvec.*VB2,2); % GradU*B*V'

% * * * * * * * * * * * * * * * * * * * * * * * *
gamma = ( - (x'*y) - tradeoff_lambda * (x_aux'*y_aux) ) / ( (y'*y) + tradeoff_lambda * (y_aux'*y_aux) );

