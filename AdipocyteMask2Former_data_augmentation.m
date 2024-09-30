addpath(genpath('C:\_research_projects\Ovarian cancer project\Ovarian_cancer\'))
imageFolder = 'C:\_research_projects\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\images intratumoral fat';
labelFolder = 'C:\_research_projects\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\masks intratumoral fat';

labelExtension = 'png';

outputFolder = 'C:\Users\wylezoln\Box\_my_projects\Ovarian cancer project\Visualizations\color augmentation\imgs'; %fullfile(imageFolder, 'Augmented2');
mkdir(outputFolder)

outputFolderMasks = 'C:\Users\wylezoln\Box\_my_projects\Ovarian cancer project\Visualizations\color augmentation\masks';%fullfile(labelFolder, 'Augmented2');
mkdir(outputFolderMasks)

files = dir([labelFolder '\*.' labelExtension]);
filesNo = length(files);
values = [-0.03, -0.025, -0.02, -0.015, -0.01, 0.01, 0.015, 0.02, 0.025, 0.03]; %to be decided
%values = [-0.05, 0.05]; 
color_num = [1:5 7 12:14];
n = numel(values);
tile_size = 1024;
%%
for i = 1:filesNo
    [~, name, ~] = fileparts(files(i).name);
    img = imread(fullfile(imageFolder, [name '.png']));
    mask = imread(fullfile(files(i).folder, files(i).name));
    for f = color_num%color_num(randperm(numel(color_num), 3))
        
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
%% IMAGE AUGMENTATION
%random color transform + random flip + random rotation + random gauss
%blurr
main_pth = "C:\_research_projects\Adipocyte model project\Mask2Former\data\training\";
image_folder = fullfile(main_pth, "images");
mask_folder = fullfile(main_pth, "_data/masks/");
save_folder_img = fullfile(main_pth, "_data/augmented images");
save_folder_mask = fullfile(main_pth, "_data/augmented masks");
mkdir(save_folder_img);
mkdir(save_folder_mask);
files = dir(image_folder);
%%
color_num = [1:5 7 12:14];
rot_opt = {'r90'; 'r180'; 'r270'; 'none'};
flip_opt = {'vflip'; 'hflip'; 'none'};
blur_opt = {'gblur';'none'};

for i = 3:size(files, 1)

file_path = fullfile(files(i).folder, files(i).name);
[~,name,~] = fileparts(file_path);

img = imread(file_path);
mask_file_path = fullfile(mask_folder, [name '.tif']);

if exist(mask_file_path, "file")
mask = imread(mask_file_path);
else
    continue;
end

rn = randi([1, numel(color_num)]);
img_aug = colorTransform(img, 'CS', rn, 'none');
rn = randi([1, numel(rot_opt)]);
[img_aug, mask_aug] = flip_rotation_blur_augmentation(img_aug, mask, rot_opt{rn});
rn = randi([1, numel(flip_opt)]);
[img_aug, mask_aug] = flip_rotation_blur_augmentation(img_aug, mask_aug, flip_opt{rn});
rn = randi([1, numel(blur_opt)]);
[img_aug, mask_aug] = flip_rotation_blur_augmentation(img_aug, mask_aug, blur_opt{rn});

imwrite(img_aug, fullfile(save_folder_img, [name '_aug.tif']));
imwrite(mask_aug, [0 0 0; 1 0 0], fullfile(save_folder_mask, [name '_aug.tif']));

end

 
% for i = 3%:size(files, 1)
%     file_path = fullfile(files(i).folder, files(i).name);
%     [~,name,~] = fileparts(file_path);
%     img = imread(file_path);
%     figure(2)
%     for j = 1:size(color_num, 2)
%         img_aug = colorTransform(img, 'CS', j, 'none');
%         %[img_aug, mask_aug] = flip_rotation_blur_augmentation(img_aug, mask_aug, 'gblur');
%         subplot(3,3, j)
% 
%         imshow(img_aug)
%     end
% 
% 
% end