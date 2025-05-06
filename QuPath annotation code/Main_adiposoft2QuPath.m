% Model to mask

folder_path = 'C:\_research_projects\Adipocyte model project\Original data\images\images blood vessels\images blood vessels';
output_path = 'C:\_research_projects\Adipocyte model project\Original data\images\images blood vessels\geojson';
mkdir(output_path);
files = dir(fullfile(folder_path, '*.tif'));

for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);

    adiposoft_mask = imread(file_path);
    FC = adiposoft2geojson(adiposoft_mask);

    fileID = fopen(fullfile(output_path, [name '.geojson']),'w');
    fwrite(fileID,jsonencode(FC, 'PrettyPrint', false));
    fclose(fileID);
end
%% correct the indexes

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