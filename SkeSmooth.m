function [P, P0, output] = SkeSmooth(Z,W,R,varargin)

%% Set parameters
params = inputParser;
params.addParameter('opt','lbfgsb', @(x) ismember(x,{'lbfgsb','ncg','tn','lbfgs'}));
params.addParameter('init', 'randn', @(x) (iscell(x) || isa(x, 'ktensor') || ismember(x,{'random','rand','randn','nvecs','zeros'})));
params.addParameter('lower',-Inf);
params.addParameter('upper',Inf);
params.addParameter('opt_options', '', @isstruct);
params.addParameter('skip_zeroing', false, @islogical);
params.addParameter('fun','auto', @(x) ismember(x,{'auto','default','sparse','sparse_lowmem'}));
params.addParameter('verbosity',10);
params.addParameter('mu', 1000);
params.parse(varargin{:});

init = params.Results.init;
opt = params.Results.opt;
options = params.Results.opt_options;
lower = params.Results.lower;
upper = params.Results.upper;
funtype = params.Results.fun;
do_zeroing = ~params.Results.skip_zeroing;
verbosity = params.Results.verbosity;
do_print = verbosity > 0;
mu = params.Results.mu;

use_lbfgsb = strcmp(opt,'lbfgsb');

if do_print
    fprintf('Running CP-WOPT...\n');
end

%% Zeroing
if do_zeroing    
    tic;
    Z = Z.*W;
    ztime = toc;
    fprintf('Time for zeroing out masked entries of data tensor is %.2e seconds.\n', ztime);
    fprintf('(If zeroing is done in preprocessing, set ''skip_zeroing'' to true.)\n');
end

%% Initialization
sz = size(Z);
N = length(sz);

if iscell(init)
    P0 = init;
elseif isa(init,'ktensor')
    P0 = tocell(init);
else
    P0 = cell(N,1);
    if strcmpi(init,'nvecs')
        for n=1:N
            P0{n} = nvecs(Z,n,R);
        end
    else
        for n=1:N
            P0{n} = matrandnorm(feval(init,sz(n),R));
        end
    end
end

%% Set up lower and upper (L-BFGS-B only)

if ~use_lbfgsb && ( any(isfinite(lower)) || any(isfinite(upper)) )
    error('Cannot use lower and upper bounds without L-BFGS-B');
end

if use_lbfgsb
    lower = convert_bound(lower,sz,R);
    upper = convert_bound(upper,sz,R);
end

%% Set up optimization algorithm

if use_lbfgsb % L-BFGS-B
    if ~exist('lbfgsb','file')
        error(['CP_OPT requires L-BFGS-B function. This can be downloaded'...
            'at https://github.com/stephenbeckr/L-BFGS-B-C']);
    end
else % POBLANO
    switch (params.Results.opt)
        case 'ncg'
            opthandle = @ncg;
        case 'tn'
            opthandle = @tn;
        case 'lbfgs'
            opthandle = @lbfgs;
    end
    
    if ~exist('poblano_params','file')
        error(['CP_OPT requires Poblano Toolbox for Matlab. This can be ' ...
            'downloaded at http://software.sandia.gov/trac/poblano.']);
    end     
end


%% Set up optimization algorithm options
if isempty(options) 
    if use_lbfgsb
        options.maxIts = 10000;
        options.maxTotalIts = 50000;
        if do_print
            options.printEvery = verbosity;
        else
            options.printEvery = 0;
        end
    else
        options = feval(opthandle, 'defaults');
    end
end

%% Set up function handle
normZsqr = norm(Z)^2;
funhandle = @(x) tt_cp_wfun(Z,W,x,normZsqr,mu,R);

%% Fit CP using CP_WOPT by ignoring missing entries

if use_lbfgsb
    opts = options;
    opts.x0 = tt_fac_to_vec(P0);    
    [xx,ff,out] = lbfgsb(funhandle, lower, upper, opts);
    P = ktensor(tt_cp_vec_to_fac(xx, Z));
    output.ExitMsg = out.lbfgs_message1;
    output.f = ff;
    output.OptOut = out;
else
    P0_vec = tt_fac_to_vec(P0);
    out = feval(opthandle, funhandle, P0_vec, options);
    P  = ktensor(tt_cp_vec_to_fac(out.X,Z));
    output.ExitFlag  = out.ExitFlag;
    output.FuncEvals = out.FuncEvals;
    output.f = out.F;
    output.G = tt_cp_vec_to_fac(out.G,W);
    output.OptOut = out;
end

%% Clean up final result
% Arrange the final tensor so that the columns are normalized.
P = arrange(P);
% Fix the signs
P = fixsigns(P);

function [f,G] = tt_cp_wfg(Z,W,A,normZsqr,mu,R) % dense W
%TT_CP_WFG Function and gradient of CP with missing data.
%
%   [F,G] = TT_CP_WFG(Z,W,A) computes the function and gradient values of
%   the function 0.5 * || W .* (Z - ktensor(A)) ||^2. The input A is a
%   cell array containing the factor matrices. The input W is a (dense
%   or sparse) tensor containing zeros wherever data is missing. The
%   input Z is a (dense or sparse) tensor that is assumed to have
%   zeros wherever there is missing data. The output is the function F
%   and a cell array G containing the partial derivatives with respect
%   to the factor matrices.
%
%   [F,G] = TT_CP_WFG(Z,W,A,NORMZSQR) also passes in the pre-computed
%   norm of Z, which makes the computations faster. 
%
%   See also TT_CP_WFUN, TT_CP_WFG_SPARSE, TT_CP_WFG_SPARSE_SETUP.
%
%MATLAB Tensor Toolbox.
%Copyright 2015, Sandia Corporation.

% This is the MATLAB Tensor Toolbox by T. Kolda, B. Bader, and others.
% http://www.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2015) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in the file LICENSE.txt


%% Compute B = W.*ktensor(A)
if isa(W,'sptensor')
    B = W.*ktensor(A);
else
    B = W.*full(ktensor(A));
end

%% Compute normZ
if ~exist('normZsqr','var')
    normZsqr = norm(Z)^2;
end

% regularization matrix
sz = size(A{1});
coeff = [-0.55,−0.19,0.04];
L = regul_mat(coeff,sz(1));

% compute the value of the regularization term
regul = mu * (norm(L*A{1}));

% function value
f = 0.5 * normZsqr - innerprod(Z,B) + 0.5 * norm(B)^2 + regul;

% gradient computation
N = ndims(Z);
G = cell(N,1);
T = Z - B;

for n = 2:3
    G{n} = zeros(size(A{n}));
    G{n} = -mttkrp(T,A,n);  % modification here to add smooth constraints
end
mtp = -mttkrp(T,A,1);
% for t = 1:length(idx)
%     ii = idx(t,:);
%     mtp(ii,:) = zeros(1,R);
% end
grad = mu * (L'*L)*A{1};
G{1} = mtp + grad;

function [f,g] = tt_cp_wfun(Zdata,W,x,normZsqr,mu,R)
%TT_CP_WFUN Computes function and gradient for weighted CP.
%
%   [F,G] = TT_CP_WFUN(Z,W,x,normZsqr) calculates the function and gradient
%   for the function 0.5 * || W .* (Z - ktensor(A)) ||^2 where W is an
%   indicator for missing data (0 = missing, 1 = present), Z is the data
%   tensor that is being fit (assumed that missing entries have already
%   been set to zero), A is a cell array of factor matrices that is created
%   from the vector x, and normZsqr in the norm of Z squared.
%
%   [F,G] = TT_CP_WFUN(Zvals,W,x,normZsqr) is a special version that takes
%   just the nonzeros in Z as calculated by the helper function
%   CP_WFG_SPARSE_SETUP.
%
%   [F,G] = TT_CP_WFUN(....,false) uses a more memory efficient version for
%   the sparse code.
%
%   See also TT_CP_WFG, TT_CP_WFG_SPARSE, TT_CP_WFG_SPARSE_SETUP
%
%MATLAB Tensor Toolbox.
%Copyright 2015, Sandia Corporation.

% This is the MATLAB Tensor Toolbox by T. Kolda, B. Bader, and others.
% http://www.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2015) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in the file LICENSE.txt


%% Convert x to factor matrices (i.e., a cell array).
A = tt_cp_vec_to_fac(x,W); 

%% Compute the function and gradient
if ~exist('normZsqr','var') %jump into this: exist normZsqr
    normZsqr = norm(Zdata)^2;
end
[f,G] = tt_cp_wfg(Zdata,W,A,normZsqr,mu,R); %dense weight matrix W


%% Convert gradient to a vector
g = tt_fac_to_vec(G); % unfold the whole 3 cell of Gradient to a vector

