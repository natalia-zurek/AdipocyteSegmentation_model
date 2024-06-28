folder_path = 'D:\Datasets\Omental metastases\selected BMI\masks\goejson old';
output_path = 'D:\Datasets\Omental metastases\selected BMI\masks\geojson';
mkdir(output_path);
files = dir(fullfile(folder_path, '*.geojson'));

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