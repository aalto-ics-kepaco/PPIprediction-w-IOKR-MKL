function [x] = cvx_pnorm_follow(a,p,n,lambda1,lambda2),

   cvx_begin quiet
    variable x(n);
    minimise(-x'*a'+lambda2*sum(power(x,p))+lambda1*sum(power(x,2)) ) ;
    subject to
      x >= 0;
   cvx_end

  
end
