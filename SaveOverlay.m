%% FROM PNG

folder_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\masks intratumoral fat';
img_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\images intratumoral fat';
output_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\images intratumoral fat\overlay';
mkdir(output_path);
files = dir(fullfile(img_path, '*.png'));
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);
    I = imread(fullfile(img_path, [name '.png']));
    mask = imread(fullfile(folder_path, [name '.png']));
    mask = bwlabel(mask,4);
    ov = labeloverlay(I, mask, "Transparency", 0.6);
    imwrite(ov, fullfile(output_path, [name '.png']));
end

%% FROM MAT
folder_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\predictions\model Ov1 MTC aug 1024 intratumoral fat\omental mets part 2';
mat_path = fullfile(folder_path, 'mat');
%img_path = "C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\test dataset\abdominal_laparoscopy\images";
%img_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\test dataset\omental mets intratumoral fat ROIs\20x';
img_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\test dataset\omental mets part 2\images';

output_path = fullfile(folder_path, 'label overlay');
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
    imwrite(ov, fullfile(output_path, [name '.png']));
end