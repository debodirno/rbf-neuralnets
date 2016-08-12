 
    totalW = 3*10;
    w = rand(size(X_train,2), totalW)-0.5;
    eta0 = 0.9;
    iter = 10000;
    etaN = eta0;
    tau2 = iter/2;
    [I,J] = ind2sub([1,3], 1:totalW);
    alpha = 0.5;
    sig0 = 15;
    sigN = sig0;
    tau1 = iter/2*log(sigN);
    numcases = size(X_train,1);    

 for i = 1:iter
    if i<iter *0.9
        etaN = eta0 * exp(-i/tau2);
        %sigN = sig0*exp(-i/tau1);
        sigN = 1;
    else
        etaN = 0.05;
        sig0 = 1;
    end
        j = randi(numcases,1,1);
        x = X_train(j,:)';
        distan = [];
        for k = 1:totalW
            z = [w(:,k)';x'];
            distan = [distan  pdist(z,'euclidean')];
        end
        clear z;
        [v,ind] = min(distan);
        ri = [I(ind), J(ind)];       
        distan = 1/(sqrt(2*pi)*sigN).*exp( sum(( ([I( : ), J( : )] - repmat(ri, totalW,1)) .^2) ,2)/(-2*sigN)) * etaN;
        for rr = 1:totalW
            w(:,rr) = w(:,rr) + distan(rr).*( x - w(:,rr));
        end
        clear distan
end
	 
output = [];
    for j = 1:numcases
        x = X_train(j,:)';
        distan = [];
        for k = 1:totalW
            z = [w(:,k)';x'];
            distan = [distan  pdist(z,'euclidean')];
        end
        clear z;
        [v,ind] = min(distan);
        output = [output ; w(:,ind)']; 
     end
    

