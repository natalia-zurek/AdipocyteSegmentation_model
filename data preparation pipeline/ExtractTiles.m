folder_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\images';
output_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\images 512 rescale';
mkdir(output_path);
files = [dir(fullfile(folder_path, '*.tif')); dir(fullfile(folder_path, '*.png'))];


ann_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\annotations';
out_ann = "C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\annotations 512 rescale";
mkdir(out_ann)
%%
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);
    I = imread(file_path);
    % tiles = extract_tiles_from_image(I, 256, 256);
    % for j = 1:size(tiles,1)
    % 
    % imwrite(tiles{j}, fullfile(output_path, [name '_' num2str(j) '.png']));
    %end

    I_res = imresize(I, 1/2, "bilinear");
    load(fullfile(ann_path, [name '.mat']));
    inst_map = imresize(inst_map, 1/2, "nearest");

    imwrite(I_res, fullfile(output_path, files(i).name));
    save(fullfile(out_ann, [name '.mat']), "inst_map");
end