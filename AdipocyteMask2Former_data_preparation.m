% ========= PREPARE DATASET FOR MASK2FORMER ========== 
org_img_path = "C:\_research_projects\Adipocyte model project\Original data\images\images Unet original";
org_mask_path = "C:\_research_projects\Adipocyte model project\Original data\masks\masks unet original";
out_folder_img = "C:\_research_projects\Adipocyte model project\Mask2Former\data\AL\images";
out_folder_mask = "C:\_research_projects\Adipocyte model project\Mask2Former\data\AL\masks";
mkdir(out_folder_img)
mkdir(out_folder_mask)

files = [dir(fullfile(org_img_path, '*.tif')); dir(fullfile(org_img_path, '*.jpg')); dir(fullfile(org_img_path, '*.png'))];
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,ext] = fileparts(file_path);
    imageName = [name ext];

    mask_path_full = fullfile(org_mask_path, [name '.png']);
    img = imread(file_path);
    class_map = imread(mask_path_full);
    
    new_img = imresize(imresize(img, 0.393, "bilinear"), [1024, 1024], "bilinear");
    new_mask = imresize(imresize(class_map, 0.393, "nearest"), [1024, 1024],"nearest");
    
    imwrite(new_mask, [0 0 0; 1 0 0], fullfile(out_folder_mask, [name '.png']));
    imwrite(new_img, fullfile(out_folder_img, [name '.png']))

end

%%
main_pth = "C:\_research_projects\Adipocyte model project\Mask2Former\data\validation";
images_path = fullfile(main_pth, "images/")';
out_folder = fullfile(main_pth, "images without mask")';
overlay_path = fullfile(main_pth, "mask overlay")';
mask_path = fullfile(main_pth, "masks/");
save_path = fullfile(main_pth, "mask2former annotations");
%mkdir(out_folder)
mkdir(overlay_path);
mkdir(save_path)

addpath(genpath('c:/Ovarian cancer project/AdipocyteSegmentation_model'));

save_overlay = 1;
save_dataset = 1;

files = [dir(fullfile(images_path, '*.tif')); dir(fullfile(images_path, '*.jpg')); dir(fullfile(images_path, '*.png'))];
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,ext] = fileparts(file_path);
    imageName = [name ext];
    
    mask_path_full = fullfile(mask_path, [name '.tif']);
    % if ~isfile(mask_path_full)
    %     disp(['Mask ' name '.png doesnt exist'])
    %     status = movefile(file_path, out_folder);
    %     continue
    % end

    img = imread(file_path);
    class_map = imread(mask_path_full);

    % prepare data
    inst_map = bwlabel(class_map, 4);
    if save_overlay
    ov = labeloverlay(img, inst_map, "Transparency", 0.6);
    imwrite(ov, fullfile(overlay_path, [name '.png']))
    end

    if save_dataset

    save(fullfile(save_path, [name '.mat']),'inst_map', 'class_map')
    end

end

%% DIVIDE DATASET INTO VALIDATION AND TRAINING
image_main_path = 'C:\_research_projects\Adipocyte model project\Mask2Former\data';
save_img_path = "C:\_research_projects\Adipocyte model project\Mask2Former\data";
mask_main_pth = "C:\_research_projects\Adipocyte model project\Mask2Former\data\binary masks";

%%
files = dir(fullfile(image_main_path, '*.tif'));
names = {files.name}.';
[val_images, train_images] = split_images(names, 0.2);
transfer_files(train_images, image_main_path, fullfile(save_img_path, "training/images"), 'move');
transfer_files(train_images, mask_main_pth, fullfile(save_img_path, "training/masks"), 'copy');
transfer_files(val_images, image_main_path, fullfile(save_img_path, "validation/images"), 'move');
transfer_files(val_images, mask_main_pth, fullfile(save_img_path, "validation/masks"), 'copy');


function [val_images, train_images] = split_images(image_files, validation_ratio)
    % Split a list of image files into validation and training sets
    % 
    % Inputs:
    %   - image_files: cell array of image file names
    %   - validation_ratio: ratio of images to be used for validation (e.g., 0.2 for 20%)
    %
    % Outputs:
    %   - val_images: cell array of validation image file names
    %   - train_images: cell array of training image file names

    num_images = length(image_files);

    num_val_images = round(validation_ratio * num_images);
    
    val_indices = randperm(num_images, num_val_images);
    
    val_images = image_files(val_indices);

    train_indices = setdiff(1:num_images, val_indices);
    train_images = image_files(train_indices);
end