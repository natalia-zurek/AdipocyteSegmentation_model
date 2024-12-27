% Model to mask

folder_path = 'C:\Users\wylezoln\Box\_my_projects\Kidney project\NataliaZ\ATmask';
output_path = 'C:\Users\wylezoln\Box\_my_projects\Kidney project\NataliaZ\geojson';

mkdir(output_path);
files = dir(fullfile(folder_path, '*.mat'));
%%
for i = 1%:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);

    load(file_path, 'ATmask');
    idx = unique(ATmask);
    % AT_mask(AT_mask == 0) = idx(end)+1;
    % AT_mask(AT_mask == -1) = 0;
    %instance_mask = maskstack2instancemask(masks);
    FC = classmask2geojson(ATmask);

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