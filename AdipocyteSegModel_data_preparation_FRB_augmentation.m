addpath(genpath('C:\Immune infiltrate project\immune infiltrate\Immune-infiltrate-project'));
imageFolder = 'C:\Ovarian cancer project\Adipocyte dataset\train\images color augmented';
labelFolder = 'C:\Ovarian cancer project\Adipocyte dataset\train\final_data_v2';

outputFolder = 'C:\Ovarian cancer project\Adipocyte dataset\train\images_all'; %fullfile(imageFolder, 'Augmented2');
mkdir(outputFolder)

outputFolderMasks = 'C:\Ovarian cancer project\Adipocyte dataset\train\final_data_v2';%fullfile(labelFolder, 'Augmented2');
mkdir(outputFolderMasks)

files = [dir(fullfile(imageFolder, '*.jpg')); dir(fullfile(imageFolder, '*.png'))];
filesNo = length(files);
min_tile_size = 1024;


aug_opt = {'r90', 'r180', 'r270', 'vflip', 'hflip', 'gblur', 'mblur' };


for i = 1:filesNo
    [~, name, ~] = fileparts(files(i).name);
    img = imread(fullfile(files(i).folder, files(i).name));

    for f = 1:size(aug_opt, 2)
        load(fullfile(labelFolder, [name '.mat']))
        mask = maskstack2instancemask(masks);
        [img_aug, mask_aug]= flip_rotation_blur_augmentation(img, mask, aug_opt{f});

        if f ~= 6 && f ~= 7
            props = regionprops(mask_aug, 'BoundingBox');
        bbox = cat(1, props.BoundingBox);
    
        end

        

        masks = instancemask2maskstack(mask_aug);

        imageName = [name '_' aug_opt{f} '.png'];
        imwrite(img_aug, fullfile(outputFolder, imageName));
        save(fullfile(outputFolderMasks, [name '_' aug_opt{f} '.mat']), 'bbox', 'masks', 'imageName', 'label')
    end
end
%%
overlayedImage = insertObjectMask(img,masks);
imshow(overlayedImage)
showShape("rectangle",boxes,Label=labels,LineColor=[1 0 0])