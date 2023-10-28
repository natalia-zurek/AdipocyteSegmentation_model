% ========= ADIPOCYTE MODEL INFERENCE ========== 

%load model
%load();
image_path = 'C:\Ovarian cancer project\omental images\tiles fat wsi 2048\12019';%C:\Ovarian cancer project\Adipocyte dataset\images MTD\';

output_path = 'C:\Ovarian cancer project\Adipocyte dataset\images MTD\inference_test_model3_1024';
mkdir(output_path);
files = [dir(fullfile(image_path, '*.tif')); dir(fullfile(image_path, '*.png')); dir(fullfile(image_path, '*.jpg'))];

for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);
    img = imread(file_path);

    [masks,labels,scores,boxes] = segmentObjects(net,img,Threshold=0.95);

    save(fullfile(output_path , [name '.mat']),'masks', 'labels', 'scores', 'boxes')

    %save overlay
    numMasks = size(masks,3);
    RGB = insertObjectMask(img, masks, MaskColor=lines(numMasks));
    imwrite(RGB, fullfile(output_path, [name '.png']));

end

