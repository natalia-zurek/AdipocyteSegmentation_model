%% non nested
folder_path = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former raw\images OM2 normal infer\mat';
img_folder = 'C:\_research_projects\Adipocyte model project\Original data\images\images OM2 1024';
output_path = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\OM2 1024';
mkdir(output_path);
post_proc = 1;
files = dir(fullfile(folder_path, '*.mat'));
%%
output_dir = fullfile(output_path, 'mat');
mkdir(output_dir)
output_dir2 = fullfile(output_path, 'masks');
mkdir(output_dir2)
output_dir3 = fullfile(output_path, 'overlays');
mkdir(output_dir3)

for i = 1:size(files, 1)
    file = files(i);
    load(fullfile(file.folder, file.name));
    [~,name,~] = fileparts(file.name);
    if isempty(inst_ids)
        inst_map(inst_map == -1) = 0;
    else
        [inst_map, inst_types, inst_ids, inst_scores] = python2matlab_instseg(inst_map, 'inst_types', inst_types, 'inst_ids', inst_ids, 'inst_scores', inst_scores);
    end
        inst_map = postproc_inst_seg(inst_map);
        I = imread(fullfile(img_folder, [name '.tif']));
        ov = labeloverlay(I, inst_map, "Transparency", 0.6);

        save(fullfile(output_dir, file.name), 'inst_map', 'inst_ids', 'inst_scores', 'inst_types', 'post_proc');
        imwrite(uint8(inst_map), fullfile(output_dir2, [name '.tif']));
        imwrite(ov, fullfile(output_dir3, [name '.png']));
end
%% nested
main_pth = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\PLCO\';
folder_path = fullfile(main_pth, 'PLCO Mask2Former raw');

img_folder = fullfile(main_pth,"PLCO tiles 1024/");
output_path = fullfile(main_pth, "PLCO Mask2Former postproc/");

mkdir(output_path);
post_proc = 1;
folders = dir(folder_path);
%%
h = waitbar(0, 'Instance segmentation postprocessing...');
for i = 36:size(folders,1)
    waitbar(i/size(folders, 1), h, sprintf('Analyzing: %d / %d', i, size(folders, 1)));
    folder = folders(i);
    if folder.isdir && ~strcmp(folder.name, '.') && ~strcmp(folder.name, '..')
        files = dir(fullfile(folder_path, folder.name, 'mat', '*.mat'));
        output_dir = fullfile(output_path, folder.name, 'mat');
        mkdir(output_dir)
        output_dir2 = fullfile(output_path, folder.name, 'masks');
        mkdir(output_dir2)
        output_dir3 = fullfile(output_path, folder.name, 'overlays');
        mkdir(output_dir3)
        for j = 1:size(files,1)

            file = files(j);
            load(fullfile(file.folder, file.name));
            [~,name,~] = fileparts(file.name);
            if isempty(inst_ids)
            inst_map(inst_map == -1) = 0;
            else
            [inst_map, inst_types, inst_ids, inst_scores] = python2matlab_instseg(inst_map, 'inst_types', inst_types, 'inst_ids', inst_ids, 'inst_scores', inst_scores);
            end

            inst_map = postproc_inst_seg(inst_map);
            I = imread(fullfile(img_folder, folder.name, [name '.tif']));
            ov = labeloverlay(I, inst_map, "Transparency", 0.6);

            save(fullfile(output_dir, file.name), 'inst_map', 'inst_ids', 'inst_scores', 'inst_types', 'inst_types', 'post_proc');
            imwrite(uint8(inst_map), fullfile(output_dir2, [name '.tif']));
            imwrite(ov, fullfile(output_dir3, [name '.png']));

        end
    end
end
close(h)

%%
function smoothed_mask = postproc_inst_seg(instance_mask)

% Perform morphological operations
smoothed_mask = zeros(size(instance_mask)); % Initialize the smoothed mask
unique_ids = unique(instance_mask);
unique_ids(unique_ids == 0) = []; % Exclude background (ID = 0)
se_size = 5;
se = strel('disk' ,se_size);
for i = 1:length(unique_ids)
    % Extract binary mask for the current instance
    single_instance = instance_mask == unique_ids(i);
    single_instance = imfill(single_instance, "holes");
    % Apply morphological opening to remove "staircases"
    opened_instance = imopen(single_instance, se);

    % Apply morphological closing to fill small gaps along edges
    smoothed_instance = imclose(opened_instance, se);

    % Add the smoothed instance back into the final mask
    smoothed_mask(smoothed_instance) = unique_ids(i);
end
end