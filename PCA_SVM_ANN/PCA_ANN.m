%% PCA + SVM
close all;
clear all;
clc;

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
label = zeros(noData,noGroup);
labels=zeros(noData,1);
for i=1:noGroup
    label(((i-1)*samplePerG+1):(i*samplePerG),i)=ones(samplePerG,1);
    labels(((i-1)*samplePerG+1):(i*samplePerG),1)=ones(samplePerG,1)*i;
end

%% Split test data and training data
noTestData = 2*noGroup;
test_g_idx=zeros(noGroup,noTestData/noGroup);
test_idx=zeros(1,noTestData);
for group=1:noGroup
    %((samp-1)*11+1):(samp*11) is goal sample; others are references.
    test_g_idx(group,:)=(group-1)*samplePerG+sort(randperm(samplePerG,noTestData/noGroup));
    test_idx(1,((group-1)*noTestData/noGroup+1):(group*noTestData/noGroup))=test_g_idx(group,:);
end
train_idx = setdiff(1:noData,test_idx);
fprintf('Finished spliting data. \n');

%% Use PCA to reduce dimensionality
threshold = 0.95;% Threshold for selection of principal components
[coeff, ~, latent]=pca(dataSet(train_idx,:));

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

%% ANN
% Create a Pattern Recognition Network
hiddenLayerSize = 2*size(dataPCA,2);
net = patternnet(hiddenLayerSize);

% Set NN parameters
net.trainParam.epochs=10000;
net.trainParam.time=10000;

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/90;
net.divideParam.valRatio = 20/90;
net.trainFcn = 'trainscg';  % Scaled conjugate gradient
net.trainParam.showWindow=0;
% Train the Network
[net,tr] = train(net,dataPCA(train_idx,:)',label(train_idx,:)');

% Test the Network
ann_prob = net(dataPCA(test_idx,:)');

% View the Network
% view(net)
% Plot confusion matrix
% figure, plotconfusion(label(test_idx,:)',ann_pred)
ann_pred=zeros(noTestData,1);
for i=1:noTestData
   ann_pred(i) = find(ann_prob(:,i)==max(ann_prob(:,i)));
end

ann_acc = sum(ann_pred==labels(test_idx))/size(ann_pred,1);
fprintf('Accuracy of ANN is %f. \n',ann_acc);

