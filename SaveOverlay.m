%% FROM PNG

folder_path = 'D:\QuPath projects\Adipocyte dataset projects\Adipocyte TCGA\adipocyte masks v2';
img_path = 'D:\QuPath projects\Adipocyte dataset projects\Adipocyte TCGA\images';
output_path = 'D:\QuPath projects\Adipocyte dataset projects\Adipocyte TCGA\overlay_v2';
mkdir(output_path);
files = dir(fullfile(img_path, '*.tif'));
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);
    I = imread(fullfile(img_path, [name '.tif']));
    mask = imread(fullfile(folder_path, [name '.png']));
    mask = bwlabel(mask,4);
    ov = labeloverlay(I, mask, "Transparency", 0.6);
    imwrite(ov, fullfile(output_path, [name '.png']));
end

%% FROM MAT
% folder_path = 'C:\_research_projects\Adipocyte model project\Mask2Former\data\training_v1';
% mat_path = fullfile(folder_path, "annotations/");
% img_path = fullfile(folder_path, "images/");
mat_path = "C:\_research_projects\Adipocyte model project\Mask2Former_v1\predictions\images student project 1024 normal infer\mat";
img_path = 'C:\_research_projects\Adipocyte model project\Original data\images\images student project 1024';
%img_path = fullfile(folder_path, '896');

output_path = fullfile('C:\_research_projects\Adipocyte model project\Mask2Former_v1\predictions\images student project 1024 normal infer\mat', 'overlays');
mkdir(output_path);
files = dir(fullfile(mat_path, '*.mat'));
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);
    I = imread(fullfile(img_path, [name '.tif']));
    load(fullfile(mat_path, [name '.mat']));
    idx = unique(inst_map);

    %inst_map = imresize(inst_map, [size(I, 1) size(I,2)], "nearest");
    inst_map(inst_map == 0) = idx(end)+1;
    inst_map(inst_map == -1) = 0;

    %mask = bwlabel(mask,4);
    ov = labeloverlay(I, inst_map, "Transparency", 0.6);
    %imshow(ov)
    imwrite(ov, fullfile(output_path, [name '.tif']));
end