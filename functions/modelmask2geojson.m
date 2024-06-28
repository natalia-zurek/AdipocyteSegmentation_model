% Model to mask

folder_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\predictions\model Ov1 MTC aug 1024\omental mets intratumoral fat 20x\mat';
output_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\predictions\model Ov1 MTC aug 1024\omental mets intratumoral fat 20x\mat\geojson';

mkdir(output_path);
files = dir(fullfile(folder_path, '*.mat'));
%%
for i = [1:3 5 11 14 25 28]%1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);

    load(file_path, 'inst_map');
    idx = unique(inst_map);
    inst_map(inst_map == 0) = idx(end)+1;
    inst_map(inst_map == -1) = 0;
    %instance_mask = maskstack2instancemask(masks);
    FC = instancemask2geojson(inst_map);

    fileID = fopen(fullfile(output_path, [name '.geojson']),'w');
    fwrite(fileID,jsonencode(FC, 'PrettyPrint',true));
    fclose(fileID);
end
%%

output_path2 = fullfile(output_path, 'corrected_geojson');
mkdir(output_path);
files = dir(fullfile(output_path, '*.geojson'));

for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);

    % Read the GeoJSON file
geojsonText = fileread(file_path);

% Replace "[1," with "[0," and " 1]" with " 0)"
modifiedGeojsonText = strrep(geojsonText, '[1,', '[0,');
modifiedGeojsonText = strrep(modifiedGeojsonText, ',1]', ',0]');


% Write the modified GeoJSON text to the output file
fileID = fopen(fullfile(output_path, files(i).name), 'w');
fprintf(fileID, '%s', modifiedGeojsonText);
fclose(fileID);

end