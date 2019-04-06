function [U_ B_ V_ RMSE_tr RMSE_te MAE_tr MAE_te] = CST(train_vec, test_vec, U0, V0, tradeoff_lambda)
% target: {1,2,3,4,5,?}, and we use {1,2,3,4,5,0} in MATLAB

MAX_EPOCH = 50;
% ---
RMSE_tr = zeros(MAX_EPOCH, 1);
RMSE_te = zeros(MAX_EPOCH, 1);
MAE_tr  = zeros(MAX_EPOCH, 1);
MAE_te  = zeros(MAX_EPOCH, 1);
% ---

%% epoch 1
% --- Initialization
U = U0;
V = V0;

% --- Estimate B
B = EstimateB(U,V,train_vec);

% --- save the parameters
U_ = U; V_ = V; B_ = B;

% --- Evaluation
i=1;
[rmse, mae] = EvalPred(U,V,B,train_vec,1,0); RMSE_tr(i) = rmse; MAE_tr(i) = mae;
[rmse, mae] = EvalPred(U,V,B,test_vec,1,0);  RMSE_te(i) = rmse; MAE_te(i) = mae;
fprintf( 1, 'epoch: %d, tr: %6.4f(RMSE), %6.4f(MAE); te: %6.4f(RMSE), %6.4f(MAE)\n', i, RMSE_tr(i), MAE_tr(i),  RMSE_te(i), MAE_te(i) );

%% epoch 2, 3, 4, ...
for i = 2 : MAX_EPOCH
  
% step 1: update U
    GradU = CalGradU(U, V, B,  train_vec,  U0, tradeoff_lambda);
	gamma = Cal_gamma(U, GradU, V, B, train_vec, U0, tradeoff_lambda);
	U = U - gamma*GradU;

% step 2: update V
    GradV = CalGradU(V, U, B', train_vec(:,[2 1 3]), V0, tradeoff_lambda);
    gamma = Cal_gamma(V, GradV, U, B', train_vec(:,[2 1 3]), V0, tradeoff_lambda);
    V = V - gamma*GradV;
    
% step 3: Estimate B
    B = EstimateB(U,V,train_vec);

    % --- Check Performance
    [rmse, mae] = EvalPred(U,V,B,train_vec,1,0); RMSE_tr(i) = rmse; MAE_tr(i) = mae;
    [rmse, mae] = EvalPred(U,V,B,test_vec,1,0);  RMSE_te(i) = rmse; MAE_te(i) = mae;
    fprintf( 1, 'epoch: %d, tr: %6.4f(RMSE), %6.4f(MAE); te: %6.4f(RMSE), %6.4f(MAE)\n', i, RMSE_tr(i), MAE_tr(i),  RMSE_te(i), MAE_te(i) );
    % ---

    % --- stopping criteria
    if i > 1
        if ( abs( RMSE_te(i-1) - RMSE_te(i) ) < 1E-4 ) || ( RMSE_te(i-1) < RMSE_te(i) )
            break;
        end
    end
    % ---
    
    % --- save the parameters
    U_ = U; V_ = V; B_ = B;
end

%% Result
RMSE_tr = RMSE_tr( RMSE_tr>0 );
RMSE_te = RMSE_te( RMSE_te>0 );
MAE_tr = MAE_tr( MAE_tr>0 );
MAE_te = MAE_te( MAE_te>0 );


%% calculate the gradient
function GradU = CalGradU(U, V, B, train_vec, U0, tradeoff_lambda)

% --- number of users and items
num_user = size(U,1);
num_item = size(V,1);

% --- user-index, item-index
uIDX    = train_vec(:,1);
vIDX    = train_vec(:,2);

% --- matrix multiplication
VB      = V*B';

% ---
U2      = U(uIDX, :);
VB2     = VB(vIDX,:);
r       = sum(U2.*VB2,2);  % U*B*V'
Diff    = train_vec(:,3)-r;   % Y .* (R - U*B*V')
P       = spconvertFULL([uIDX, vIDX, Diff], num_user, num_item);  % time !!

% ---
Pu      = P * VB;

% ---
delU    = U-U0;  % regularization term

% --- Gradient of the objective function
Gu      = - Pu + tradeoff_lambda/num_user*delU;

% --- obtain Qu to project the gradient to the tangent space at U
Qu      = -U'*Gu;

% --- final gradient
GradU   = Gu + U*Qu;


%% Calculate gamma
function gamma = Cal_gamma( U, GradU, V, B, train_vec, U0, tradeoff_lambda )

% --- number of users
num_user = size(U,1);

% --- user-index, item-index
uIDX    = train_vec(:,1);
vIDX    = train_vec(:,2);

% ---
VB      = V*B';   % V*B'

% ---
VB2     = VB(vIDX,:);

% --- 
U2      = U(uIDX,:);
r       = sum(U2.*VB2,2); % U*B*V'

% ---
Gvec    = GradU(uIDX,:);
rG      = sum(Gvec.*VB2,2); % GradU*B*V'

% ---
gamma   = ( -(train_vec(:,3)-r)'*rG + tradeoff_lambda/num_user*trace((U-U0)'*GradU) ) / ( (rG'*rG) + tradeoff_lambda/num_user*norm(GradU, 'fro') );

