folder_path = 'C:\_research_projects\Adipocyte model project\Mask2Former\data\validation\images x20';
output_path = 'C:\_research_projects\Adipocyte model project\Mask2Former\data\validation\images';
mkdir(output_path);
files = dir(fullfile(folder_path, '*.tif'));
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);
    I = imread(file_path);
    I = imresize(I, [512 512], "bicubic");
    imwrite(I, fullfile(output_path, [name '.tif']));
end
%%
folder_path = 'C:\_research_projects\Adipocyte model project\Mask2Former\data\training\annotations x20';
output_path = 'C:\_research_projects\Adipocyte model project\Mask2Former\data\training\annotations';
mkdir(output_path);
files = dir(fullfile(folder_path, '*.mat'));
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);
    load(file_path);
    class_map = imresize(class_map, [512 512], "nearest");
    inst_map = imresize(inst_map, [512 512], "nearest");
    save(fullfile(output_path, [name '.mat']), "inst_map", "class_map");
end