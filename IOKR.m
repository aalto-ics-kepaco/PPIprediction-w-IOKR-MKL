function IOKR(KKAll,labels,trset,tsset),
  % Input-Output Kernel Regression for transfer learning
  % inputs: %%%%%%%%%%%%%%%%%%%%
  %   KKAll: linear kernel on the input features concatenated for both species
  %   labels: adjacency matrix NproteinsxNproteins where 1 indicates an interactions and 0 otherwise, needs to be same size as KKAll, if there are no labels available for species 2 use zeros
  %   trset: indexes of proteins from species 1 that are used for training, because for cross validating the results you want to use only a part ie 90%
  %   tsset: indexes of proteins from species 2, here typically all are used
  
  % compute KKAll as follows %%%%%%%%%%%%%%
  % (1) for each data source (aid,bid give start and stop index of ds variable) in the feature matrix NproteinsxNfeatures
  %       feat_cur = [featuresSp1(1:size(trset,1),aid:bid); featuresSp2(:,aid:bid)];
  % (2) compute linear kernel
  %      KKsingle(:,:,ds) = feat_cur*feat_cur';
  %      KKsingle(:,:,ds) = KKsingle(:,:,ds)./ (sqrt(diag(KKsingle(:,:,ds)))*sqrt(diag(KKsingle(:,:,ds)))' + 0.00000001);
  % (3) combination of kernels of the different data sources
  %     * either naive 
  %       KKAll = sum(KKsingle,3);
  %       KKAll = KKAll./(sqrt(diag(KKAll))*sqrt(diag(KKAll))' + 0.00000001); 
  %     * or use MKL() to compute a correlation score for each data source
  %       for ds=1:size(countsScerv,2),
  %         KKsingle(:,:,ds) = KKsingle(:,:,ds).*rcorr(ds);
  %       end;
  %       KKAll = sum(KKsingle,3);
  %       KKAll = KKAll./(sqrt(diag(KKAll))*sqrt(diag(KKAll))' + 0.00000001); 

  % parameters
  lambda1=0.7; % or cross validate
  lambda2 = 0.0005; % or cross validate
  Beta = 1;
  
  labels_aux = labels(1:size(trset,1),1:size(trset,1));
  % Laplacian unnormalized
  L = diag(sum(labels_aux)) - labels_aux;
  % Difussion output kernel matrix
  Diff_Kernel = expm(-Beta*L);
  % Normalize
  Diff_Kernel =  Diff_Kernel ./ (sqrt(diag(Diff_Kernel)) * sqrt(diag(Diff_Kernel))'); 

  if strcmp(setup{tio},'supIOKR')
     A = supervised_setting(size(trset,1), KKAll, lambda1);
  else, 
     A = semi_supervised_setting(size(trset,1), size(KKAll,1), KKAll, lambda1, lambda2, Beta); 
  end;
         
  % Predictions
  predictions = A' * Diff_Kernel * A;
  
  ranktrtr = predictions(1:size(trset,1),1:size(trset,1));
  ranktsts = predictions(size(trset,1)+1:end,size(trset,1)+1:end);
  
  labelstrtr = labels(1:size(trset,1),1:size(trset,1));
  labelststs = labels(size(trset,1)+1:end,size(trset,1)+1:end);
  
  % compute AUC statistic, ROC curve and PR curve
  [AUCTRTR, ROCTR, PRTR] = getAUCandROCandPR(labelstrtr(:),ranktrtr(:));
  [AUCTSTS, ROCTS, PRTS] = getAUCandROCandPR(labelststs(:),ranktsts(:));
  
  
  
  % SUPERVISED setting
  %%%%%%%%%%%%%%%%%%%%%%%
  function [A] = supervised_setting(Strset, KKAll, lambda1),
    B = lambda1 * eye(Strset,Strset) + KKAll(1:Strset,1:Strset);
    A = B \ KKAll(1:Strset,:);
  end

  % SEMI-SUPERVISED setting
  %%%%%%%%%%%%%%%%%%%%%%%%%%%
  function [A] = semi_supervised_setting(Strset, Slabels, KKAll, lambda1, lambda2, Beta),
      
    U = zeros(Strset, Slabels);
    U(:,1:Strset) = eye(Strset);

    LKKAll = diag(sum(KKAll)) - KKAll;
    LKKAll = expm(-Beta * LKKAll);
    
    % Normalize matrix
    LKKAll = LKKAll ./ (sqrt(diag(LKKAll)) * sqrt(diag(LKKAll))');

    B = U/(lambda1 * eye(Slabels) + KKAll * (U'*U) + 2*lambda2*KKAll*LKKAll);
    A = B * KKAll;
    
  end
  
  % help functions
  
  function [AUC,ROC,PR,PrAPos] = getAUCandROCandPR(labels,predictions),  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    
    TPR = 0;
    FPR = 0;
    
    TPRprev = 0;
    FPRprev = 0;
    AUC = 0;
    bthresh = 0.5;

    if size(predictions,1)>20000,
      predictions = quant(predictions,0.001);
    end;

    %min(predictions)
    %max(predictions)

    [pred_sort,idsort] = sort(predictions,'descend');
    labelsort = labels(idsort);
    
  
    %if size(labelsort,1)>10000,
    %  [pred_sort(1:50)    labelsort(1:50)]
     
    %  size(find(labelsort(1:2000)))
    %  size(find(labelsort(1:10000)))
    %end;

    if size(labelsort,1)>=3000,
    PrAPos = size(find(labelsort(1:3000)),1);
    else,
     PrAPos = 0;
    end;

    Pos = size(find(labels==1),1);
    if Pos==0, 
      'ppiFrame: Warning - no positive examples!'
    end;

    Neg = size(find(labels==0),1); 
    if Neg==0, 
      'ppiFrame: Warning - no negative examples!'
    end; 

    %[pred_sort labelsort]

    i=1;
    lprev = -1000;
    ROC = []; 
    PR = [];
    minDst = 2;

    while i<=size(pred_sort,1),

    if pred_sort(i)~=lprev,
     if Pos>0,
      ROC = [ROC; FPR/Neg TPR/Pos];
      PR = [PR; TPR/Pos TPR/(TPR+FPR)];
      Dst = sqrt((0-FPR/Neg)^2 + (1-TPR/Pos)^2);
     else,
      PR = [PR; 1 TPR/(TPR+FPR)];
      ROC = [ROC; FPR/Neg 1];
      Dst = sqrt((0-FPR/Neg)^2 + (1-1)^2);
     end;
     
     if minDst>Dst,
       minDst=Dst;
       bthresh = pred_sort(i);
     end;

     AUC = AUC + calcarea(FPR,FPRprev,TPR,TPRprev);

     lprev = pred_sort(i);
     TPRprev = TPR;
     FPRprev = FPR;
    end;

    if labelsort(i)==1,
     TPR = TPR+1;
    else
     FPR = FPR+1;
    end;

    i = i+1;
    end;

    if Pos>0,
     ROC = [ROC; FPR/Neg TPR/Pos];
     PR = [PR; TPR/Pos TPR/(TPR+FPR)];
    else,
     PR = [PR; 1 TPR/(TPR+FPR)];
     ROC = [ROC; FPR/Neg 1];
    end;
    AUC = AUC + calcarea(FPR,FPRprev,TPR,TPRprev);
    if Pos>0,
    AUC = AUC/(Pos * Neg);
    else,
    AUC = FPR/Neg;
    end;

    
  end

  function A=calcarea(X1,X2,Y1,Y2), %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    base = abs(X1-X2);
    height = (Y1+Y2)/2;
    A = base*height;
  end


end