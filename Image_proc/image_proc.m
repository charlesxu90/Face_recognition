close all;
clear all
clc

% Load Datasets

Dataset = '/Users/charles/Documents/Programs/MATLAB/MachineLearning/Image_proc/Train';   
Testset  = '/Users/charles/Documents/Programs/MATLAB/MachineLearning/Image_proc/Test';


% we need to process the images first.
% Convert your images into grayscale
% Resize the images

width=100; height=100;
DataSet   = cell([], 1);

%% Preprocessing(including formatting, resizing)
 for i=1:length(dir(fullfile(Dataset,'*.jpg')))

     % Training set process
     k = dir(fullfile(Dataset,'*.jpg'));
     k = {k(~[k.isdir]).name};
     for j=1:length(k)
        tempImage  = imread(horzcat(Dataset,filesep,k{j}));
        imgInfo    = imfinfo(horzcat(Dataset,filesep,k{j}));

         % Image transformation
         if strcmp(imgInfo.ColorType,'grayscale')
            DataSet{j} = double(imresize(tempImage,[width height])); % array of images
         else
            DataSet{j} = double(imresize(rgb2gray(tempImage),[width height])); % array of images
         end
     end
end

TestSet =  cell([], 1);
  for i=1:length(dir(fullfile(Testset,'*.jpg')))

     % Training set process
     k = dir(fullfile(Testset,'*.jpg'));
     k = {k(~[k.isdir]).name};
     for j=1:length(k)
        tempImage   = imread(horzcat(Testset,filesep,k{j}));
        imgInfo     = imfinfo(horzcat(Testset,filesep,k{j}));

         % Image transformation
         if strcmp(imgInfo.ColorType,'grayscale')
            TestSet{j}  = double(imresize(tempImage,[width height])); % array of images
         else
            TestSet{j}  = double(imresize(rgb2gray(tempImage),[width height])); % array of images
         end
     end
end

% Prepare class label for first run of svm
% I have arranged labels 1 & 2 as per my convenience.
% It is always better to label your images numerically
% Please note that for every image in our Dataset we need to provide one label.
% we have 30 images and we divided it into two label groups here.
train_label               = zeros(size(30,1),1);
train_label(1:15,1)   = 1;         % 1 = Airplanes
train_label(16:30,1)  = 2;         % 2 = Faces

% Prepare numeric matrix for svmtrain
Training_Set=[];
for i=1:length(DataSet)
    Training_Set_tmp   = reshape(DataSet{i},1, 100*100);
    Training_Set=[Training_Set;Training_Set_tmp];
end

Test_Set=[];
for j=1:length(TestSet)
    Test_set_tmp   = reshape(TestSet{j},1, 100*100);
    Test_Set=[Test_Set;Test_set_tmp];
end

% Perform first run of svm
SVMStruct = svmtrain(Training_Set , train_label, 'kernel_function', 'linear');
Group       = svmclassify(SVMStruct, Test_Set);