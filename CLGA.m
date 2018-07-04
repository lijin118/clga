function [Z,A] = CLGA(Xs,Xt,Ys,Yt0,options,Goptions)


if ~isfield(options,'k')
    options.k = 100;
end
if ~isfield(options,'lambda')
    options.lambda = 0.1;
end
if ~isfield(options,'ker')
    options.ker = 'primal';
end
if ~isfield(options,'gamma')
    options.gamma = 1.0;
end
if ~isfield(options,'data')
    options.data = 'default';
end
k = options.k;
lambda = options.lambda;
ker = options.ker;
gamma = options.gamma;
data = options.data;


X = [Xs,Xt];
X = X*diag(sparse(1./sqrt(sum(X.^2))));
[m,n] = size(X);
ns = size(Xs,2);
nt = size(Xt,2);
C = length(unique(Ys));

% Construct MMD matrix
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = e*e'*C;
if ~isempty(Yt0) && length(Yt0)==nt
    for c = reshape(unique(Ys),1,C)
        e = zeros(n,1);
        e(Ys==c) = 1/length(find(Ys==c));
        e(ns+find(Yt0==c)) = -1/length(find(Yt0==c));
        e(isinf(e)) = 0;
        M = M + e*e';
    end
end
M = M/norm(M,'fro');

% Construct centering matrix
H = eye(n)-1/(n)*ones(n,n);


%Construct Graph
if strcmp(Goptions.model,'sig')
    %Goptions.gnd=[Ys;Yt0];       
    W= constructW(X',Goptions);
    D = diag(full(sum(W,2)));
    W1 = -W;
    for i=1:size(W1,1)
        W1(i,i) = W1(i,i) + D(i,i);
    end
else
     [Ww,Wb]=ConG(Goptions.gnd_train,Goptions,X','srge');     
     W1=Ww-options.gamma*Wb;
end

if strcmp(ker,'primal')
    [A,~] = eigs(X*M*X'+lambda*eye(m)+X*(options.beta*W1)*X',X*H*X',k,'SM');        
    Z = A'*X;
    
else
    K = kernel(ker,X,[],gamma);
    [A,~] = eigs(K*M*K'+lambda*eye(n)+options.beta*W1,K*H*K',k,'SM');    
    Z = A'*K;
   
end


end
