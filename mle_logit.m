    function rmsd = mle_logit(b,x,y)
 p= 1./(1+exp(-b(1)*(x-b(2))));

rmsd = -1*sum(y .* log(p) + (1 - y) .* log(1 - p));
