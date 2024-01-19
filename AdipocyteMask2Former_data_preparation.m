% ========= PREPARE DATASET FOR MASK2FORMER ========== 
main_pth = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset';
images_path = fullfile(main_pth, "images")';
out_folder = fullfile(main_pth, "images without mask")';
overlay_path = fullfile(main_pth, "mask overlay")';
mask_path = fullfile(main_pth, "augmented masks");
save_path = fullfile(main_pth, "annotations");
mkdir(out_folder)
mkdir(overlay_path);
mkdir(save_path)

addpath(genpath('c:/Ovarian cancer project/AdipocyteSegmentation_model'));

save_overlay = 0;
save_dataset = 1;

files = [dir(fullfile(images_path, '*.tif')); dir(fullfile(images_path, '*.jpg')); dir(fullfile(images_path, '*.png'))];
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,ext] = fileparts(file_path);
    imageName = [name ext];
    
    mask_path_full = fullfile(mask_path, [name '.png']);
    if ~isfile(mask_path_full)
        disp(['Mask ' name '.png doesnt exist'])
        status = movefile(file_path, out_folder);
        continue
    end

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