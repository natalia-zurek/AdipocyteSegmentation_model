folder_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\to change';
mask_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\to change\masks';
output_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\images new\';
output_path_m = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\images new\masks';
mkdir(output_path_m);
files = dir(fullfile(folder_path, '*.png'));
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);
    img = imread(file_path);
    mask = imread(fullfile(mask_path, [name '.png']));
    if size(img, 1) == 1024
    imwrite(img, fullfile(output_path, [name '.tif']));
    imwrite(mask, fullfile(output_path_m, [name '.png']));
    else
    upper_left = img(1:1024, 1:1024, :);
    upper_right = img(1:1024, end-1023:end, :);
    lower_left = img(end-1023:end, 1:1024, :);
    lower_right = img(end-1023:end, end-1023:end, :);
    % subplot(2,2,1)
    % imshow(upper_left)
    % subplot(2,2,2)
    % imshow(upper_right)
    % subplot(2,2,3)
    % imshow(lower_left)
    % subplot(2,2,4)
    % imshow(lower_right)
    imwrite(upper_left, fullfile(output_path, [name '_ul.tif']));
    imwrite(upper_right, fullfile(output_path, [name '_ur.tif']));
    imwrite(lower_left, fullfile(output_path, [name '_ll.tif']));
    imwrite(lower_right, fullfile(output_path, [name '_lr.tif']));
    
    mask = imread(fullfile(mask_path, [name '.png']));

    upper_left = mask(1:1024, 1:1024);
    upper_right = mask(1:1024, end-1023:end);
    lower_left = mask(end-1023:end, 1:1024);
    lower_right = mask(end-1023:end, end-1023:end);

        imwrite(upper_left, fullfile(output_path_m, [name '_ul.png']));
    imwrite(upper_right, fullfile(output_path_m, [name '_ur.png']));
    imwrite(lower_left, fullfile(output_path_m, [name '_ll.png']));
    imwrite(lower_right, fullfile(output_path_m, [name '_lr.png']));
    end
    
end
%%
test = imread('C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\training dataset\images new\masks\0-11900_GTEX-13QJC_Adipose-Subcutaneous_ll.png');
