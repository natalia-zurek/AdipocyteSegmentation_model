addpath(genpath('C:\Immune infiltrate project\immune infiltrate\Immune-infiltrate-project'));
imageFolder = 'C:\Ovarian cancer project\Adipocyte dataset\train\images';
labelFolder = 'C:\Ovarian cancer project\Adipocyte dataset\train\final_data_v2';

outputFolder = 'C:\Ovarian cancer project\Adipocyte dataset\train\images color augmented'; %fullfile(imageFolder, 'Augmented2');
mkdir(outputFolder)

outputFolderMasks = 'C:\Ovarian cancer project\Adipocyte dataset\train\final_data_v2';%fullfile(labelFolder, 'Augmented2');
mkdir(outputFolderMasks)

files = [dir(fullfile(imageFolder, '*.jpg')); dir(fullfile(imageFolder, '*.png'))];
filesNo = length(files);
min_tile_size = 1024;

for i = 1:filesNo
    [~, name, ~] = fileparts(files(i).name);
    img = imread(fullfile(files(i).folder, files(i).name));

    for f = [1:5 7 12:14]
        load(fullfile(labelFolder, [name '.mat']))
        rImg = colorTransform(img, 'CS', f, 'none');
        imageName = [name '_' num2str(f) '.png'];
        imwrite(rImg, fullfile(outputFolder, imageName));
        save(fullfile(outputFolderMasks, [name '_' num2str(f) '.mat']), 'bbox', 'masks', 'imageName', 'label')
    end
end
