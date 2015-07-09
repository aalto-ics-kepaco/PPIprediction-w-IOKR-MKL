function rcorr = MKL(KKsingle,yy,Combtype,trset),
   % multiple kernel learning, returns a correlation score for each of the data sources depending how well they align with the output labels
   % Note: the computation uses only the data from the training set
   % inputs: %%%%%%%%%%%%%%%%%%
   %   KKsingle: linear kernel on the input features concatenated for both species, one for each data source
   %    compute KKsingle as follows %%%%%%%%%%%%%%
   %    (1) for each data source (aid,bid give start and stop index of ds variable) in the feature matrix NproteinsxNfeatures
   %       feat_cur = [featuresSp1(1:size(trset,1),aid:bid); featuresSp2(:,aid:bid)];
   %    (2) compute linear kernel
   %      KKsingle(:,:,ds) = feat_cur*feat_cur';
   %      KKsingle(:,:,ds) = KKsingle(:,:,ds)./ (sqrt(diag(KKsingle(:,:,ds)))*sqrt(diag(KKsingle(:,:,ds)))' + 0.00000001);
   %   yy: adjacency matrix NproteinsxNproteins where 1 indicates an interactions and 0 otherwise, needs to be same size as KKAll, if there are no labels available for species 2 use zeros
   %   Combtype: 'rcorr' centered kernel target alignment on individual kernels
   %             'rcorrComb'  combined centered kernel target alignment, also known as alignf
   %             'rcorrCombpnormfollow' pnorm path following           
   %   trset: indexes of proteins from species 1 that are used for training, because for cross validating the results you want to use only a part ie 90%

   
   if strcmp(Combtype, 'rcorr') % might be possible ..........
     % individual alignment of kernels and labels
     for ds=1:size(countsScerv,2), % over all data sources
         % centering and correlation computation
         rcorr(ds) = computeCenteredCorr(KKsingle([1:size(trset,1)],[1:size(trset,1)],ds),yy);
     end;
   end; 

   % combined optimization methods - alignf
   if strcmp(Combtype, 'rcorrComb') 
       rcorr = computeCenteredCorrCombined(KKsingle([1:size(trset,1)],[1:size(trset,1)],:),yy,1,0);
   end;
   
   % pnorm path following - always needs a grid search to determine the best lambdas ie  l = [0.001 0.01 0.1 1];
   if strcmp(Combtype, 'rcorrCombpnormfollow') 
       rcorr = computeCenteredCorrCombined(KKsingle([1:size(trset,1)],[1:size(trset,1)],:),yy,3,[l(i1) l(i2)]); 
   end;
   
   

   % centering and correlation computation %%%%%%%%%%%%%%%%%%
   function rcorr = computeCenteredCorr(Kern,Label)
   
    KKAllC =  (eye(size(Kern,1))-ones(size(Kern))./size(Kern,1))*Kern*(eye(size(Kern,1))-ones(size(Kern))./size(Kern,1)); 
    LabelC =  (eye(size(Kern,1))-ones(size(Kern))./size(Kern,1))*Label*(eye(size(Kern,1))-ones(size(Kern))./size(Kern,1)); 

    if norm(KKAllC,'fro')>0,
     rcorr = sum(diag(KKAllC'*LabelC))/(norm(KKAllC,'fro')*norm(LabelC,'fro'));
    else,
     rcorr = 0;
    end;

  end;
  
  % centering and combined correlation computation %%%
  function [rcorrAdj] = computeCenteredCorrCombined(Kern,Label,cvxon,lam)
   
   M = [];
   LabelC =  (eye(size(Label,1))-ones(size(Label))./size(Label,1))*Label*(eye(size(Label,1))-ones(size(Label))./size(Label,1)); 
   
   for si=1:size(Kern,3),
     K_cur1 = Kern(:,:,si);
     K_cur1 =  (eye(size(K_cur1,1))-ones(size(K_cur1))./size(K_cur1,1))*K_cur1*(eye(size(K_cur1,1))-ones(size(K_cur1))./size(K_cur1,1)); 
      
     if cvxon<2, % pnorm does not need M
     for ti=si:size(Kern,3),
    
         K_cur2 = Kern(:,:,ti);
         K_cur2 =  (eye(size(K_cur2,1))-ones(size(K_cur2))./size(K_cur2,1))*K_cur2*(eye(size(K_cur2,1))-ones(size(K_cur2))./size(K_cur2,1)); 
         
         M(si,ti) = sum(diag( K_cur1'*K_cur2 ));
         M(ti,si) = M(si,ti); % symmetric
     end; 
     end;
     a2(si) = sum(diag(K_cur1'*LabelC));

   end;

   M = M +0.0001; % for numerical stability  
   
   switch cvxon,
    case 3 % pnorm path following
      p = 2.0;
      dp = 0.01;
      epsi = exp(-1); 

      % inital optimization
      n = size(a2,2);
      
      [x] = cvx_pnorm_follow(a2,p,n,lam(1),lam(2));
      
      % predictor-corrector algo
      while p>1.5, 
       % p-step
       p = p -dp;
       x(find(x<epsi)) = 0;
       x = x - dp *(-lam(2)*x.^(p-1).*(1+p*log(x+0.00001))) ./ (lam(1)*2 + lam(2)*p*(p-1)*(x+0.00001).^(p-2));
      
       % c-step
       ff = -x.*a2'+lam(2)*x.^p +lam(1)*sum(power(x,2))/size(x,1); % 
       dff = -a2' + lam(2)*p*(x).^(p-1)+ lam(1)*2*sum(x)/size(x,1);  % 
       x = x - dp * ff./(dff);
      end;
      
      x(find(x<epsi)) = 0;
      rcorrAdj = x./norm(x,1);
   
      
    case 1  % alignf
      n = size(Kern,3)
      [x] = cvx_test(a2,M,n);

      rcorrAdj = x./norm(x,1);
      
    
   end
   
  end 


end