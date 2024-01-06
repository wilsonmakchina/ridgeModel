function [Vm,a] =  crossValModel_relative_contribution(cR, Vc, cBeta, opt)
% function to compute cross-validated R^2, 
% let certain regression coefficients to be zero in order to measure its
% importance

folds = opt.folds;
cv_seq = opt.cv_seq;

nt = size(cR,1);

Vm = zeros(size(Vc),'single'); %pre-allocate motor-reconstructed V
foldCnt = floor(size(Vc,2) / folds);

rng(1) % for reproducibility
if cv_seq == 1
    randIdx = randperm(size(Vc,2)); %generate randum number index
else
    if cv_seq == 2
        DT = 50;%floor(foldCnt/folds_sub);
    elseif cv_seq == 3
        DT = 500;%floor(foldCnt/folds_sub);
    end
    randIdx = 1:DT:nt; % the start index of folds in test set
    if randIdx(end)+DT > nt
        randIdx = randIdx(1:end-1);   
    end
end
randIdx = shuffle(randIdx);

for iFolds = 1:folds
    dataIdx = true(1,size(Vc,2));
    
    if cv_seq ~= 1
        for i = 1+(iFolds-1):folds:length(randIdx)
            dataIdx(randIdx(i):randIdx(i)+DT-1) = false; 
            tmp = dataIdx;
        end
        a{iFolds} = dataIdx;
    else
        dataIdx(randIdx(((iFolds - 1)*foldCnt) + (1:foldCnt))) = false; %index for training data
    end

    Vm(:,~dataIdx) = ([ones(sum(~dataIdx), 1) cR(~dataIdx,:)] * cBeta{iFolds})'; %predict remaining data

    if rem(iFolds,folds/5) == 0
        fprintf(1, 'Current fold is %d out of %d\n', iFolds, folds);
    end
end


end