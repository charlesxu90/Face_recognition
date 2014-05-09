%% PCA + SVM
close all;
clear all;
clc;
mtimes=10;
threshold = 0.80;% Threshold for selection of principal components
%% Load images
% As yaleface data is already grayscale and fixed size(243X320)
% So I can just use it directly
imgDir = '../pgm';
% imgDir = '../backup/Data/supported';
olddir = pwd; % Save current folder
chdir(imgDir); % Change to faces folder
directory_list = dir('*.pgm'); % Read directory list
% get rid of . and ..
directory_list = directory_list(~[directory_list.isdir]);
chdir(olddir);
% Get file size
Img=pgmRead(['../pgm/' directory_list(1).name]);
[width, height] = size(Img);
dataSet=zeros(length(directory_list), width*height);

for i=1:length(directory_list)
    %   fprintf('reading yalefaces/%s \n', directory_list(i).name);
    Img=pgmRead(['../pgm/' directory_list(i).name]);
    dataSet(i,:) = reshape(Img, 1, width*height);
end
fprintf('Finished reading data. \n');
noData=size(dataSet,1);
noGroup = 15;
samplePerG = noData/noGroup;
% Add labels
labels = zeros(noData,noGroup);
label = zeros(noData,1);
for i=1:noGroup
    label(((i-1)*samplePerG+1):(i*samplePerG),1)=ones(samplePerG,1)*i;
    labels(((i-1)*samplePerG+1):(i*samplePerG),i)=ones(samplePerG,1);
end

%% Split test data and training data
noTestData = 2*noGroup;
test_g_idx=zeros(noGroup,noTestData/noGroup);
test_idx=zeros(1,noTestData);
testNO=0;
while testNO<mtimes
    for group=1:noGroup
        %((samp-1)*11+1):(samp*11) is goal sample; others are references.
        
        test_g_idx(group,:)=(group-1)*samplePerG+sort(randperm(samplePerG,noTestData/noGroup));
        test_idx(1,((group-1)*noTestData/noGroup+1):(group*noTestData/noGroup))=test_g_idx(group,:);
    end
    train_idx = setdiff(1:noData,test_idx);
    fprintf('Finished spliting data. \n');
    
    %% Use PCA to reduce dimensionality
    try
        [coeff, ~, latent]=pca(dataSet(train_idx,:));
    catch err
        continue;
    end
    testNO=testNO+1;
    dataSet(:,:)=bsxfun(@minus, dataSet(:,:),mean(dataSet(train_idx,:),1));
    fprintf('Finished norm data. \n');
    % Select No. of principal components
    i = 0;
    eigval_sum=0;
    while eigval_sum < threshold*sum(latent)
        i=i+1;
        eigval_sum = eigval_sum + latent(i);
    end
    dataPCA=dataSet*coeff(:,1:i);
    fprintf('Finished PCA feature selection. \n');
    
    %% SVM
    % Using one-versus-all method
    % add path to the libsvm toolbox
    addpath('./libsvm/matlab');
    
    d1SVM=cell(noGroup,1);
    d1prob=zeros(noTestData,noGroup);
    d2SVM=cell(noGroup,1);
    d2prob=zeros(noTestData,noGroup);
    d5SVM=cell(noGroup,1);
    d5prob=zeros(noTestData,noGroup);
    d10SVM=cell(noGroup,1);
    d10prob=zeros(noTestData,noGroup);
%     d50SVM=cell(noGroup,1);
%     d50prob=zeros(noTestData,noGroup);
    for group=1:noGroup
        % Degree 1 Poly SVM
        d1SVM{group} = svmtrain(double(label(train_idx)==group),dataPCA(train_idx,:), ...
            '-t 0 -b 1 -c 10 -h 0');
        [~, ~, p] = svmpredict(double(label(test_idx)==group),dataPCA(test_idx,:), d1SVM{group}, '-b 1');
        d1prob(:,group) = p(:,d1SVM{group}.Label==1);
        
        % Degree 2 Poly SVM
        d2SVM{group} = svmtrain(double(label(train_idx)==group),dataPCA(train_idx,:), ...
            '-t 1 -b 1 -c 1 -d 3 -r 10 -h 0');
        [~, ~, p] = svmpredict(double(label(test_idx)==group),dataPCA(test_idx,:), d2SVM{group}, '-b 1');
        d2prob(:,group) = p(:,d2SVM{group}.Label==1);
        
        % Degree 5 Poly SVM
        d5SVM{group} = svmtrain(double(label(train_idx)==group),dataPCA(train_idx,:), ...
            '-t 2 -b 1 -c 10 -d 5 -h 0');
        [~, ~, p] = svmpredict(double(label(test_idx)==group),dataPCA(test_idx,:), d5SVM{group}, '-b 1');
        d5prob(:,group) = p(:,d5SVM{group}.Label==1);
        
        % Degree 10 Poly SVM
        d10SVM{group} = svmtrain(double(label(train_idx)==group),dataPCA(train_idx,:), ...
            '-t 2 -b 1 -c 10 -d 5 -h 0');
        [~, ~, p] = svmpredict(double(label(test_idx)==group),dataPCA(test_idx,:), d10SVM{group}, '-b 1');
        d10prob(:,group) = p(:,d10SVM{group}.Label==1);
        
%         % Degree 50 Poly SVM
%         d50SVM{group} = svmtrain(double(label(train_idx)==group),dataPCA(train_idx,:), ...
%             '-t 2 -b 1 -c 10 -d 5 -h 0');
%         [~, ~, p] = svmpredict(double(label(test_idx)==group),dataPCA(test_idx,:), d50SVM{group}, '-b 1');
%         d50prob(:,group) = p(:,d50SVM{group}.Label==1);
%         
    end
    
    d1pred=zeros(noTestData,1);
    d2pred=zeros(noTestData,1);
    d5pred=zeros(noTestData,1);
    d10pred=zeros(noTestData,1);
%     d50pred=zeros(noTestData,1);
    
    for i=1:noTestData
        [~,d1pred(i)] = max(d1prob(i,:));
        [~,d2pred(i)] = max(d2prob(i,:));
        [~,d5pred(i)] = max(d5prob(i,:));
        [~,d10pred(i)] = max(d10prob(i,:));
%         [~,d50pred(i)] = max(d50prob(i,:));
    end
    
    d1acc(testNO) = sum(d1pred==label(test_idx))/size(test_idx,2);
    d2acc(testNO) = sum(d2pred==label(test_idx))/size(test_idx,2);
    d5acc(testNO) = sum(d5pred==label(test_idx))/size(test_idx,2);
    d10acc(testNO) = sum(d10pred==label(test_idx))/size(test_idx,2);
%     d50acc(testNO) = sum(d50pred==label(test_idx))/size(test_idx,2);
    
end

%% Print result
fprintf('Accuracy of SVM using Degree 1 Polynomial is %f, std %f. \n',mean(d1acc),std(d1acc));
fprintf('Accuracy of SVM using Degree 2 Polynomial is %f, std %f. \n',mean(d2acc),std(d2acc));
fprintf('Accuracy of SVM using Degree 5 Polynomial is %f, std %f. \n',mean(d5acc),std(d5acc));
fprintf('Accuracy of SVM using Degree 10 Polynomial is %f, std %f. \n',mean(d10acc),std(d10acc));
% fprintf('Accuracy of SVM using Degree 50 Polynomial is %f, std %f. \n',mean(d50acc),std(d50acc));