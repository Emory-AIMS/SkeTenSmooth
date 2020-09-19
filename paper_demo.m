%% test WOPT on different sampling.

clc; close all; clearvars;
addpath(genpath('./tensor_toolbox'));
addpath(genpath('./poblano_toolbox_1.1'));
addpath(genpath('./lbfgs'));

R=5;
mu=1;

%% load data
f = sprintf('data/oriX.mat');
S = load(f);
oriX = S.oriX;
f = sprintf('data/adaX.mat');
S = load(f);
adaX = S.adaX;
f = sprintf('data/adaW.mat');
S = load(f);
adaW = S.adaW;

%% Get ncg defaults
ncg_opts = ncg('defaults');
ncg_opts.StopTol = 1.0e-6;
ncg_opts.RelFuncTol = 1.0e-20;
ncg_opts.MaxIters = 10^4;
ncg_opts.DisplayIters = 10;

%% Create initial guess using 'nvecs'
M_init = create_guess('Data', oriX, 'Num_Factors', R, ...
    'Factor_Generator', 'nvecs');

M_init_ada = create_guess('Data', adaX, 'Num_Factors', R, ...
    'Factor_Generator', 'nvecs');

%% CP-OPT
oriX = double(oriX);
oriX = tensor(oriX);
[oriM,~,~] = cp_opt(oriX, R, 'init', M_init);

%% SkeSmooth
[adaM,~,~] = SkeSmooth(adaX, adaW, R, 'init', M_init_ada, ...
        'opt', 'ncg', 'opt_options', ncg_opts,'mu',mu);  

%% plot results
a = oriM{1};
b = adaM_perm{1};
plot(a(:,1),'linewidth',2);
hold on;
plot(b(:,1),'linewidth',2);
hold on;
hl = legend('Original','Adaptive Sampling');
set(hl,'Box','off', 'Fontsize',18,'linewidth',30);
hold off;

%% compute FMS
[scr,adaM_perm,~,~] = score(adaM,oriM,'greedy',true);

%% compute TCS
onetensor=tenones(dim);
adaM=full(adaM_perm);
adaM = oriX.*adaW + adaM.*(onetensor-adaW);
upper=oriX-adaM;
lower=oriX;
tcs=norm(upper)/norm(lower);
    