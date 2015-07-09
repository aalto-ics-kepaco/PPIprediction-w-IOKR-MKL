function [x] = cvx_test(a,M,n),


   cvx_begin quiet
    variable x(n);
    minimise((x'*M*x)-2*x'*a') ;
    subject to
      x >= 0;
   cvx_end

 
 
end
