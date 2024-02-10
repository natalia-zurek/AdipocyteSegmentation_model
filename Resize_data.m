folder_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\test dataset\omental mets intratumoral fat ROIs\40x';
output_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\test dataset\omental mets intratumoral fat ROIs\10x';
mkdir(output_path);
files = dir(fullfile(folder_path, '*.png'));
for i = 1:size(files, 1)

    file_path = fullfile(files(i).folder, files(i).name);
    I = imread(file_path);
    I = imresize(I, [512 512], "bilinear");
    imwrite(I  ,fullfile(output_path, files(i).name));

end