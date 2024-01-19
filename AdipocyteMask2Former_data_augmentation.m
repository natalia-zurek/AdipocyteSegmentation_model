addpath(genpath('C:\Immune infiltrate project\immune infiltrate\Immune-infiltrate-project'));
imageFolder = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\images';
labelFolder = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\masks';

labelExtension = 'png';

outputFolder = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\augmented images'; %fullfile(imageFolder, 'Augmented2');
mkdir(outputFolder)

outputFolderMasks = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\augmented masks';%fullfile(labelFolder, 'Augmented2');
mkdir(outputFolderMasks)

files = dir([labelFolder '\*.' labelExtension]);
filesNo = length(files);
values = [-0.03, -0.025, -0.02, -0.015, -0.01, 0.01, 0.015, 0.02, 0.025, 0.03]; %to be decided
%values = [-0.05, 0.05]; 
color_num = [1:5 7 12:14];
n = numel(values);
tile_size = 1024;

for i = 1:filesNo
    [~, name, ~] = fileparts(files(i).name);
    img = imread(fullfile(imageFolder, [name '.tif']));
    mask = imread(fullfile(files(i).folder, files(i).name));
    for f = color_num(randperm(numel(color_num), 3))
        
        rImg = colorTransform(img, 'CS', f, 'none');
        
        %resize image and mask
        num = values(randi(n));
        rImg = imresize(rImg, 1+num, "bilinear");
        aug_mask = imresize(mask, 1+num, "nearest");

        %make the image and mask match the tile_size
        if num < 0 %mirror padding
            pad_rows = tile_size - size(rImg, 1);
            pad_cols = tile_size - size(rImg, 2);
            rImg = padarray(rImg, [pad_rows, pad_cols], 'symmetric', 'post');
            aug_mask = padarray(aug_mask, [pad_rows, pad_cols], 'symmetric', 'post');

        else %crop
            rImg = imcrop(rImg, [1 1 tile_size-1 tile_size-1]);
            aug_mask = imcrop(aug_mask, [1 1 tile_size-1 tile_size-1]);
        end


        imwrite(rImg, fullfile(outputFolder, [name '_' num2str(f) '.png']))
        imwrite(aug_mask, [0 0 0; 0.5020 0 0], fullfile(outputFolderMasks, [name '_' num2str(f) '.png']))
    end
end
