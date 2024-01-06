function [Vm, cBeta, cR, subIdx, cRidge, cLabels] =  crossValModel(fullR, Vc, cLabels, regIdx, regLabels, opt)
% function to compute cross-validated R^2

folds = opt.folds;
cv_seq = opt.cv_seq;

cIdx = ismember(regIdx, find(ismember(regLabels,cLabels))); %get index for task regressors
cLabels = regLabels(sort(find(ismember(regLabels,cLabels)))); %make sure motorLabels is in the right order

%create new regressor index that matches motor labels
subIdx = regIdx;
subIdx = subIdx(cIdx);
temp = unique(subIdx);
for x = 1 : length(temp)
    subIdx(subIdx == temp(x)) = x;
end
% cR = fullR(:,cIdx);
cR = fullR;
% shuffle other variables across time
id_tmp = find(~cIdx);
nt = size(cR,1);
for i = 1:length(id_tmp)
    cR(:,id_tmp(i)) = cR(randperm(nt),id_tmp(i));
end

Vm = zeros(size(Vc),'single'); %pre-allocate motor-reconstructed V
foldCnt = floor(size(Vc,2) / folds);

rng(1) % for reproducibility
if cv_seq == 1
    randIdx = randperm(size(Vc,2)); %generate randum number index
else
    DT = cv_seq;
%     if cv_seq == 2
%         DT = 10;%floor(foldCnt/folds_sub);
%     elseif cv_seq == 3
%         DT = 20;%floor(foldCnt/folds_sub);
%     end
    randIdx = 1:DT:nt; % the start index of folds in test set
    if randIdx(end)+DT > nt
        randIdx = randIdx(1:end-1);   
    end
end
randIdx = shuffle(randIdx);
cBeta = cell(1,folds);

for iFolds = 1:folds
    dataIdx = true(1,size(Vc,2));
    
    if folds > 1
        if cv_seq ~= 1
            for i = 1+(iFolds-1):folds:length(randIdx)
                dataIdx(randIdx(i):randIdx(i)+DT-1) = false; 
            end
        else
            dataIdx(randIdx(((iFolds - 1)*foldCnt) + (1:foldCnt))) = false; %index for training data
        end
%             dataIdx(((iFolds - 1)*foldCnt) + (1:foldCnt)) = false;
        if iFolds == 1
            [cRidge, cBeta{iFolds}] = ridgeMML(Vc(:,dataIdx)', cR(dataIdx,:), false); %get beta weights and ridge penalty for task only model
        else
            [~, cBeta{iFolds}] = ridgeMML(Vc(:,dataIdx)', cR(dataIdx,:), false, cRidge); %get beta weights for task only model. ridge value should be the same as in the first run.
        end
        
        Vm(:,~dataIdx) = ([ones(sum(~dataIdx), 1) cR(~dataIdx,:)] * cBeta{iFolds})'; %predict remaining data
        
        if rem(iFolds,folds/5) == 0
            fprintf(1, 'Current fold is %d out of %d\n', iFolds, folds);
        end
    else
        [cRidge, cBeta{iFolds}] = ridgeMML(Vc', cR, false); %get beta weights for task-only model.
        Vm = ([ones(nt,1) cR] * cBeta{iFolds})'; %predict remaining data
        disp('Ridgefold is <= 1, fit to complete dataset instead');
    end
end

% % computed all predicted variance
% Vc = reshape(Vc,size(Vc,1),[]);
% Vm = reshape(Vm,size(Vm,1),[]);
% if length(size(U)) == 3
%     U = arrayShrink(U, squeeze(isnan(U(:,:,1))));
% end
% covVc = cov(Vc');  % S x S
% covVm = cov(Vm');  % S x S
% cCovV = bsxfun(@minus, Vm, mean(Vm,2)) * Vc' / (size(Vc, 2) - 1);  % S x S
% covP = sum((U * cCovV) .* U, 2)';  % 1 x P
% varP1 = sum((U * covVc) .* U, 2)';  % 1 x P
% varP2 = sum((U * covVm) .* U, 2)';  % 1 x P
% stdPxPy = varP1 .^ 0.5 .* varP2 .^ 0.5; % 1 x P
% cMap = gather((covP ./ stdPxPy)');
% 
% % movie for predicted variance
% cMovie = zeros(size(U,1),frames, 'single');
% for iFrames = 1:frames
%     
%     frameIdx = iFrames:frames:size(Vc,2); %index for the same frame in each trial
%     cData = bsxfun(@minus, Vc(:,frameIdx), mean(Vc(:,frameIdx),2));
%     cModel = bsxfun(@minus, Vm(:,frameIdx), mean(Vm(:,frameIdx),2));
%     covVc = cov(cData');  % S x S
%     covVm = cov(cModel');  % S x S
%     cCovV = cModel * cData' / (length(frameIdx) - 1);  % S x S
%     covP = sum((U * cCovV) .* U, 2)';  % 1 x P
%     varP1 = sum((U * covVc) .* U, 2)';  % 1 x P
%     varP2 = sum((U * covVm) .* U, 2)';  % 1 x P
%     stdPxPy = varP1 .^ 0.5 .* varP2 .^ 0.5; % 1 x P
%     cMovie(:,iFrames) = gather(covP ./ stdPxPy)';
%     clear cData cModel
%     
% end
% fprintf('Run finished. RMSE: %f\n', median(cMovie(:).^2));

end