%% ---- DeepLab v3+ Semantic Segmentation Model Training Script ----
% Clear workspace
clear; clc; close all;

%% ---- set variables ----
% Define dataset paths
imageDir_train = 'path\to\training\images';          % Directory containing input images
labelDir_train = 'path\to\training\annotations';     % Directory containing pixel-wise labels (ground truth)
maskDir_train = 'path\to\training\masks';

% validation path 
imageDir_valid = 'path\to\validation\images';        % Directory containing input images
labelDir_valid = 'path\to\validation\annotations';   % Directory containing pixel-wise labels (ground truth)
maskDir_valid = 'path\to\validation\masks';

classNames = ["background", "adipocyte"];            % Define class names
labelIDs = [0, 1];                                   % Pixel values corresponding to class names
colormap = [0 0 0; 1 0 0];

imageSize = [1024, 1024, 3];                         % Input image size

%% Step 1: Data Preparation
mkdir(maskDir_train);
files = dir(fullfile(labelDir_train, '*.mat'));
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);
    load(file_path);
    imwrite(class_map, colormap, fullfile(maskDir_train, [name '.png']));
end

mkdir(maskDir_valid);
files = dir(fullfile(labelDir_valid, '*.mat'));
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);
    load(file_path);
    imwrite(class_map, colormap, fullfile(maskDir_valid, [name '.png']));
end

% Load dataset
imds_train = imageDatastore(imageDir_train);
imds_valid = imageDatastore(imageDir_valid);

pxds_train = pixelLabelDatastore(maskDir_train, classNames, labelIDs);
pxds_valid = pixelLabelDatastore(maskDir_valid, classNames, labelIDs);

% Visualize dataset (optional)
idx = 20;
sampleImage = readimage(imds_train, idx);
sampleLabel = readimage(pxds_train, idx);
figure;
imshow(labeloverlay(sampleImage, sampleLabel, 'Colormap', colormap));
title('Sample Image with Ground Truth');

%% Step 2: Data Augmentation

augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', @()randi([0,3],1)*90 );

% Create augmented datastores for training
dsTrain = pixelLabelImageDatastore(imds_train, pxds_train, ...
    'DataAugmentation', augmenter);

% Validation datastore (no augmentation)
dsVal = pixelLabelImageDatastore(imds_valid, pxds_valid);

%% Step 3: Model Definition
% Load pre-trained DeepLab v3+ network with ResNet-18 backbone
numClasses = numel(classNames);
lgraph = deeplabv3plusLayers(imageSize, numClasses, 'resnet18');

%% Step 4: Training Options

    % 'ValidationData', dsVal, ...
    % 'ValidationFrequency', 50, ...
    % 'LearnRateSchedule', "piecewise", ...
    % 'LearnRateDropFactor', 0.2, ...
    % 'LearnRateDropPeriod', 10
    
% Define training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 5e-5, ...
    'ValidationData', dsVal, ...
    'ValidationFrequency', 50, ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'auto');

%% Step 5: Train & Save the Model
disp('Training started...');
[net, info] = trainNetwork(dsTrain, lgraph, options);

modelSavePath = fullfile('C:\_research_projects\Adipocyte model project\MATLAB seg\DeepLabV3+\trained models', 'DL3plus_adipocyte_Ov1_MTC_aug_1024.mat');
save(modelSavePath, 'net', "info");
disp(['Model saved to: ', modelSavePath]);