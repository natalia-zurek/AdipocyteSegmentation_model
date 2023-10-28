% Model to mask

folder_path = 'C:\Ovarian cancer project\Adipocyte dataset\images MTD\inference_test_model3_1024';
output_path = 'C:\Ovarian cancer project\Adipocyte dataset\images MTD\inference_test_model3_1024\geojson';

mkdir(output_path);
files = dir(fullfile(folder_path, '*.mat'));
for i = 21:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);

    load(file_path, 'masks');
    instance_mask = maskstack2instancemask(masks);
    FC = instancemask2geojson(instance_mask);

    fileID = fopen(fullfile(output_path, [name '.geojson']),'w');
    fwrite(fileID,jsonencode(FC, 'PrettyPrint',true));
    fclose(fileID);
end
