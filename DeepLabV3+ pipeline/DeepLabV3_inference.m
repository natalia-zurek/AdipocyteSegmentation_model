% model_path = 'path\to\model\model.mat';
% image_path = 'path\to\images';
% output_path = 'path\to\output\folder';

model_path = 'C:\_research_projects\Adipocyte model project\MATLAB seg\DeepLabV3+\trained models\DL3plus_adipocyte_Ov1_MTC_aug_1024.mat';
image_path = 'C:\_research_projects\Adipocyte model project\Original data\images\images TCGA 1024';
output_path = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\DLV TCGA new';
mkdir(output_path)

load(model_path);

mode = 'both'; %'raw' %'post' %'both'

%% non-nested folders
if strcmp(mode, 'both')
    raw_dir = fullfile(output_path, 'raw');
    post_dir = fullfile(output_path, 'post');
    if ~exist(raw_dir, 'dir'), mkdir(raw_dir); end
    if ~exist(post_dir, 'dir'), mkdir(post_dir); end

end

imds = imageDatastore(image_path);
for j = 1:numel(imds.Files)
    [~,name,~] = fileparts(imds.Files{j});
    image = readimage(imds, j);
    cat_mask = semanticseg(image, net);
    mask = uint8(cat_mask)-1;
    switch mode
        case 'raw'
            mask = bwlabel(mask);
            ov = labeloverlay(image, mask, 'Transparency', 0.6);
            save(fullfile(output_path, [name '.mat']), 'mask');
            imwrite(ov, fullfile(output_path, [name '.png']));
        case 'post'
            mask = postproc_sem_seg(mask);
            ov = labeloverlay(image, mask, 'Transparency', 0.6);
            save(fullfile(output_path, [name '.mat']), 'mask');
            imwrite(ov, fullfile(output_path, [name '.png']));
        case 'both'
            % Raw
            mask = bwlabel(mask);
            raw_ov = labeloverlay(image, mask, 'Transparency', 0.6);
            save(fullfile(raw_dir, [name '.mat']), 'mask');
            imwrite(raw_ov, fullfile(raw_dir, [name '.png']));
            % Postprocessed
            mask = uint8(cat_mask) - 1;
            mask = postproc_sem_seg(mask);
            post_ov = labeloverlay(image, mask, 'Transparency', 0.6);
            save(fullfile(post_dir, [name '.mat']), 'mask');
            imwrite(post_ov, fullfile(post_dir, [name '.png']));
        otherwise
            disp('Wrong mode!')
            break;
    end
end
disp("Done!")

%% nested folders

folders = dir(image_path);
for i = 1:size(folders,1)
    folder = folders(i);
    if folder.isdir && ~strcmp(folder.name, '.') && ~strcmp(folder.name, '..')
        out_path = fullfile(output_path, folder.name);
        mkdir(out_path)
        imds = imageDatastore(fullfile(image_path, folder.name));
        % Pre-create output directories if needed
        if strcmp(mode, 'both')
            raw_dir = fullfile(output_path, 'raw');
            post_dir = fullfile(output_path, 'post');
            if ~exist(raw_dir, 'dir'), mkdir(raw_dir); end
            if ~exist(post_dir, 'dir'), mkdir(post_dir); end

        end

        for j = 1:numel(imds.Files)
            [~, name, ~] = fileparts(imds.Files{j});
            image = readimage(imds, j);
            cat_mask = semanticseg(image, net);
            mask = uint8(cat_mask) - 1;
            switch mode
                case 'raw'
                    mask = bwlabel(mask);
                    ov = labeloverlay(image, mask, 'Transparency', 0.6);
                    save(fullfile(output_path, [name '.mat']), 'mask');
                    imwrite(ov, fullfile(output_path, [name '.png']));
                case 'post'
                    mask = postproc_sem_seg(mask);
                    ov = labeloverlay(image, mask, 'Transparency', 0.6);
                    save(fullfile(output_path, [name '.mat']), 'mask');
                    imwrite(ov, fullfile(output_path, [name '.png']));
                case 'both'
                    % Raw
                    mask = bwlabel(mask);
                    raw_ov = labeloverlay(image, mask, 'Transparency', 0.6);
                    save(fullfile(raw_dir, [name '.mat']), 'mask');
                    imwrite(raw_ov, fullfile(raw_dir, [name '.png']));
                    % Postprocessed
                    mask = uint8(cat_mask) - 1;
                    mask = postproc_sem_seg(mask);
                    post_ov = labeloverlay(image, mask, 'Transparency', 0.6);
                    save(fullfile(post_dir, [name '.mat']), 'mask');
                    imwrite(post_ov, fullfile(post_dir, [name '.png']));
                otherwise
                    disp('Wrong mode!')
                    break;

            end
        end

        disp("Done!")

    end
end

%% functions
function instance_segmentation = postproc_sem_seg(mask)

% se_size = 5;
% se = strel('disk', se_size);

mask = imfill(mask, "holes");
% Apply morphological opening to remove "staircases"
% mask = imopen(mask, se);
%
% % Apply morphological closing to fill small gaps along edges
% mask = imclose(mask, se);

inst_mask = bwlabel(mask);

%delete small objects
props = regionprops(inst_mask, 'Area');
areas = [props.Area]; % Extract the areas of the regions
small_objects = find(areas < 50); % Find objects with Area < 50px

% Create a mask with only the objects to keep
for i = 1:length(small_objects)
    inst_mask(inst_mask == small_objects(i)) = 0;
end

% Relabel the remaining objects
inst_mask = bwlabel(inst_mask);

%fill holes
inst_mask = imfill(inst_mask, "holes");

distance_transform = -bwdist(~inst_mask);
mask_with_minima = imhmin(distance_transform, 5); % Add minima to suppress unwanted regions
watershed_labels = watershed(mask_with_minima);
instance_segmentation = watershed_labels;
instance_segmentation(~inst_mask) = 0;
end