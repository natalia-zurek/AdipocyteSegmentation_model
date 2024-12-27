% ========= PREPARE DATASET FOR MASK2FORMER ========== 
org_img_path = "C:\_research_projects\Adipocyte model project\Original data\images\images student project";
org_mask_path = "C:\_research_projects\Adipocyte model project\Original data\masks\masks student project";
out_folder_img = "C:\_research_projects\Adipocyte model project\Original data\images\images student project 1024";
out_folder_mask = "C:\_research_projects\Adipocyte model project\Original data\masks\masks student project 1024";
mkdir(out_folder_img)
mkdir(out_folder_mask)
%% change pixel size of the images

files = [dir(fullfile(org_img_path, '*.tif')); dir(fullfile(org_img_path, '*.jpg')); dir(fullfile(org_img_path, '*.png'))];
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,ext] = fileparts(file_path);
    imageName = [name ext];

    mask_path_full = fullfile(org_mask_path, [name '.png']);
    img = imread(file_path);
    class_map = imread(mask_path_full);
    
    new_img = imresize(img, [1024, 1024], "bilinear");
    new_mask = imresize(class_map, [1024, 1024],"nearest");
    
    imwrite(new_mask, [0 0 0; 1 0 0], fullfile(out_folder_mask, [name '.tif']));
    imwrite(new_img, fullfile(out_folder_img, [name '.tif']))

end

%% save data into mask2former format
%% ======= PRZETESTOWAC CZY TEN KOD DZIALA Z USUWANIEM MALYCH OBIEKTOW!!!!!11 =========
main_pth = "C:\_research_projects\Adipocyte model project\Original data";
images_path = fullfile(main_pth, "images/images TCGA 1024/");
out_folder = fullfile(main_pth, "images without mask")';
overlay_path = fullfile(main_pth, "masks/masks TCGA 1024/mask overlay")';
mask_path = fullfile(main_pth, "masks/masks TCGA 1024/");
save_path = fullfile(main_pth, "annotations/annotations TCGA 1024");

% % main_pth = "C:\_research_projects\Adipocyte model project\Mask2Former\data\training\_data";
% % images_path = fullfile(main_pth, "augmented images");
% % overlay_path = fullfile(main_pth, "mask overlay")';
% % mask_path = fullfile(main_pth, "augmented masks");
% % save_path = fullfile(main_pth, "augmented annotations");

%mkdir(out_folder)
mkdir(overlay_path);
mkdir(save_path)

min_area_threshold = 10;
save_overlay = 1;
save_dataset = 1;

files = [dir(fullfile(images_path, '*.tif')); dir(fullfile(images_path, '*.jpg')); dir(fullfile(images_path, '*.png'))];
%%
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,ext] = fileparts(file_path);
    imageName = [name ext];
    
    mask_path_full = fullfile(mask_path, [name '.png']);
    % if ~isfile(mask_path_full)
    %     disp(['Mask ' name '.png doesnt exist'])
    %     status = movefile(file_path, out_folder);
    %     continue
    % end

    img = imread(file_path);
    class_map = imread(mask_path_full);

    % prepare data
    inst_map = bwlabel(class_map, 4);
    props = regionprops("struct", inst_map, 'BoundingBox', 'Area', 'PixelIdxList');
    large_objects = find([props.Area] >= min_area_threshold);
    
    for idx = setdiff(1:numel(props), large_objects)
        disp(idx)
        class_map(props(idx).PixelIdxList) = 0;  % Set the corresponding pixels in class_map to 0 (or background)
        inst_map(props(idx).PixelIdxList) = 0;
    end

    if save_overlay
    ov = labeloverlay(img, inst_map, "Transparency", 0.6);
    imwrite(ov, fullfile(overlay_path, [name '.tif']))
    end

    if save_dataset

    save(fullfile(save_path, [name '.mat']),'inst_map', 'class_map')
    end
end

%% DIVIDE DATASET INTO VALIDATION AND TRAINING
image_main_path = 'C:\_research_projects\Adipocyte model project\Original data\images\images GTEX 1024';
save_img_path = "C:\_research_projects\Adipocyte model project\Mask2Former\data";
annotation_main_pth = "C:\_research_projects\Adipocyte model project\Original data\annotations\annotations unet GTEX 1024";
mask_main_pth = "C:\_research_projects\Adipocyte model project\Original data\masks\masks unet GTEX 1024";

%%
files = dir(fullfile(image_main_path, '*.tif'));
names = {files.name}.';
%%
[val_images, train_images] = split_images(names, 0.2);

%%
val_mask = strrep(val_images, '.tif', '.png');
train_mask = strrep(train_images, '.tif', '.png');

val_anno = strrep(val_images, '.tif', '.mat');
train_anno = strrep(train_images, '.tif', '.mat');
%%
transfer_files(train_images, image_main_path, fullfile(save_img_path, "training/images"), 'copy');
transfer_files(val_images, image_main_path, fullfile(save_img_path, "validation/images"), 'copy');
%%
transfer_files(train_mask, mask_main_pth, fullfile(save_img_path, "training/masks"), 'copy');
%transfer_files(val_mask, mask_main_pth, fullfile(save_img_path, "validation/masks"), 'copy');
%%
transfer_files(train_anno, mask_main_pth, fullfile(save_img_path, "training/annotations"), 'copy');
transfer_files(val_anno, mask_main_pth, fullfile(save_img_path, "validation/annotations"), 'copy');
%%
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