folder_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\images new\masks';
img_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\images new';
output_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\images new\overlay';
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