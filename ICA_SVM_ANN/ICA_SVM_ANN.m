%% PCA + SVM
close all;
clear all;
clc;
mtimes=10;
trainModel = 4; % Choose classifers:  0.all? 1.linear SVM, 2.poly SVM, 3.rbf SVM
% 4.ANN.
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

testNO=0;
while testNO < mtimes
    %% Split test data and training data
    noTestData = 2*noGroup;
    test_g_idx=zeros(noGroup,noTestData/noGroup);
    test_idx=zeros(1,noTestData);
    val_g_idx=zeros(noGroup,noTestData/noGroup);
    val_idx=zeros(1,noTestData);

    for group=1:noGroup
        %((samp-1)*11+1):(samp*11) is goal sample; others are references.
        test_val=randperm(samplePerG,4);
        test_g_idx(group,:)=(group-1)*samplePerG+test_val(1:2);
        val_g_idx(group,:)=(group-1)*samplePerG+test_val(3:4);
        
        val_idx(1,((group-1)*2+1):(group*2))=val_g_idx(group,:);
        test_idx(1,((group-1)*2+1):(group*2))=test_g_idx(group,:);
    end
    train_idx = setdiff(1:noData,[test_idx val_idx]);
    fprintf('Finished spliting data. \n');
    
    %% Use PCA to reduce dimensionality
    threshold = 1.00;% Threshold for selection of principal components
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
    
    %% Use ICA to reduce dimensionality
    % add path to fastICA directory
    addpath('./FastICA_25');
    [IC, mixW, sepW]=fastica(dataPCA, 'g','tanh', 'approach', 'symm','displayInterval',10);%,'numOfIC', 30
    % size(IC),size(mixW),size(sepW)
    % max(max(abs(mixW*IC-dataPCA))) % Test for Mix
    % max(max(abs(IC-sepW*dataPCA))) % Test for Sep
    % Here mixW are used for learning in ANN
    dataICA=mixW;
    
    %% SVM
    % Using one-versus-all method
    % add path to the libsvm toolbox
    addpath('./libsvm/matlab');
    
    if trainModel==1 || trainModel==0
        lnr_SVM=cell(noGroup,1);
        lnr_val_prob=zeros(noTestData,noGroup);
        lnr_prob=zeros(noTestData,noGroup);
    end
    if trainModel==2 || trainModel==0
        poly_SVM=cell(noGroup,1);
        poly_val_prob=zeros(noTestData,noGroup);
        poly_prob=zeros(noTestData,noGroup);
    end
    if trainModel==3 || trainModel==0
        rbf_SVM=cell(noGroup,1);
        rbf_val_prob=zeros(noTestData,noGroup);
        rbf_prob=zeros(noTestData,noGroup);
    end
            
    for group=1:noGroup
        if trainModel==1 || trainModel==0
            % Linear SVM
            lnr_SVM{group} = svmtrain(double(label(train_idx)==group),dataICA(train_idx,:), ...
                '-t 0 -b 1 -c 10 -h 0');
            % Make a p rediction for validation data
            [~, ~, p] = svmpredict(double(label(val_idx)==group),dataICA(val_idx,:), lnr_SVM{group}, '-b 1');
            lnr_val_prob(:,group) = p(:,lnr_SVM{group}.Label==1);
            if trainModel==0
                % Make a p rediction for test data
                [~, ~, p] = svmpredict(double(label(test_idx)==group),dataICA(test_idx,:), lnr_SVM{group}, '-b 1');
                lnr_prob(:,group) = p(:,lnr_SVM{group}.Label==1);
            end
        end
        if trainModel==2 || trainModel==0
            % Polynomial SVM
            poly_SVM{group} = svmtrain(double(label(train_idx)==group),dataICA(train_idx,:), ...
                '-t 1 -b 1 -c 10 -d 1 -r 5 -h 0');
            % Make a p rediction for validation data
            [~, ~, p] = svmpredict(double(label(val_idx)==group),dataICA(val_idx,:), poly_SVM{group}, '-b 1');
            poly_val_prob(:,group) = p(:,poly_SVM{group}.Label==1);
            if trainModel==0
                % Make a p rediction for test data
                [~, ~, p] = svmpredict(double(label(test_idx)==group),dataICA(test_idx,:), poly_SVM{group}, '-b 1');
                poly_prob(:,group) = p(:,poly_SVM{group}.Label==1);
            end
        end
        if trainModel==3 || trainModel==0
            % RBF SVM
            rbf_SVM{group} = svmtrain(double(label(train_idx)==group),dataICA(train_idx,:), ...
                '-t 2 -b 1 -d 1 -h 0');
            % Make a p rediction for validation data
            [~, ~, p] = svmpredict(double(label(val_idx)==group),dataICA(val_idx,:), rbf_SVM{group}, '-b 1');
            rbf_val_prob(:,group) = p(:,rbf_SVM{group}.Label==1);
            if trainModel==0
                % Make a prediction for test data
                [~, ~, p] = svmpredict(double(label(test_idx)==group),dataICA(test_idx,:), rbf_SVM{group}, '-b 1');
                rbf_prob(:,group) = p(:,rbf_SVM{group}.Label==1);
            end
        end
    end
    
    lnr_val_pred=zeros(noTestData,1);
    poly_val_pred=zeros(noTestData,1);
    rbf_val_pred=zeros(noTestData,1);
    lnr_pred=zeros(noTestData,1);
    poly_pred=zeros(noTestData,1);
    rbf_pred=zeros(noTestData,1);
    for i=1:noTestData
        if trainModel==1 || trainModel==0
            [~,lnr_val_pred(i)] = max(lnr_val_prob(i,:));
            [~,lnr_pred(i)] = max(lnr_prob(i,:));
        end
        if trainModel==2 || trainModel==0
            [~,poly_val_pred(i)] = max(poly_val_prob(i,:));
            [~,poly_pred(i)] = max(poly_prob(i,:));
        end
        if trainModel==3 || trainModel==0
            [~,rbf_val_pred(i)] = max(rbf_val_prob(i,:));
            [~,rbf_pred(i)] = max(rbf_prob(i,:));
        end
    end
    
    if trainModel==1
        lnr_val_acc(testNO) = sum(lnr_val_pred==label(val_idx))/size(val_idx,2);   
    elseif trainModel==2
        poly_val_acc(testNO) = sum(poly_val_pred==label(val_idx))/size(val_idx,2);
    elseif trainModel==3
        rbf_val_acc(testNO) = sum(rbf_val_pred==label(val_idx))/size(val_idx,2);
    end
    if trainModel==0
        lnr_acc(testNO) = sum(lnr_pred==label(test_idx))/size(test_idx,2);
        poly_acc(testNO) = sum(poly_pred==label(test_idx))/size(test_idx,2);
        rbf_acc(testNO) = sum(rbf_pred==label(test_idx))/size(test_idx,2);
    end
    
    %% ANN
    if trainModel==4 || trainModel==0
        % Create a Pattern Recognition Network
        hiddenLayerSize = 2*size(dataICA,2);
        net = patternnet(hiddenLayerSize);
        % Set NN parameters
        net.trainParam.epochs=100000;
%         net.trainParam.time=10000;
        % Setup Division of Data for Training, Validation, Testing
        net.divideParam.trainRatio = 70/90;
        net.divideParam.valRatio = 20/90;
        net.trainFcn = 'trainscg';  % Scaled conjugate gradient
        net.trainParam.showWindow=0;
        
        % Train the Network
        [net,tr] = train(net,dataICA([train_idx val_idx],:)',labels([train_idx val_idx],:)');
        % Test the Network
        ann_prob = net(dataICA(test_idx,:)');
        
        % Plot confusion matrix
        % figure, plotconfusion(labels(test_idx,:)',ann_pred)
        ann_pred=zeros(noTestData,1);
        for i=1:noTestData
            ann_pred(i) = find(ann_prob(:,i)==max(ann_prob(:,i)));
        end
        ann_acc(testNO) = sum(ann_pred==label(test_idx))/size(ann_pred,1);
    end
    fprintf('%d iteration done...\n',testNO);
end

%% Print result
if trainModel==1
    fprintf('Accuracy of Linear SVM is %f, std %f, max %f. \n',mean(lnr_val_acc),std(lnr_val_acc),max(lnr_val_acc));
elseif trainModel==2
    fprintf('Accuracy of Polynomial SVM is %f, std %f,max %f. \n',mean(poly_val_acc),std(poly_val_acc),max(poly_val_acc));
elseif trainModel==3
    fprintf('Accuracy of RBF SVM is %f, std %f,max %f. \n',mean(rbf_val_acc),std(rbf_val_acc),max(rbf_val_acc));
end

if trainModel==0
    fprintf('Accuracy of Linear SVM is %f, std %f, max %f. \n',mean(lnr_acc),std(lnr_acc),max(lnr_acc));
    fprintf('Accuracy of Polynomial SVM is %f, std %f,max %f. \n',mean(poly_acc),std(poly_acc),max(poly_acc));
    fprintf('Accuracy of RBF SVM is %f, std %f,max %f. \n',mean(rbf_acc),std(rbf_acc),max(rbf_acc));
    fprintf('Accuracy of ANN is %f, std %f,max %f. \n',mean(ann_acc),std(ann_acc),max(ann_acc));
end