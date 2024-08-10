main_pth = "C:\Users\wylezoln\Box\_my_projects\Ovarian cancer project\Adipocyte segmentation model\datasets";
mask_pth = fullfile(main_pth, "masks/masks TCGA/");
image_path = fullfile(main_pth, "images/images TCGA/");
output_path = 'C:\_research_projects\Ovarian cancer project';
% mkdir(output_path);
dataset_name = 'TCGA';
files = dir(fullfile(mask_pth, '*.png'));

T = table();
for i = 1:size(files, 1)
    file_name = files(i).name;
    file_path = fullfile(files(i).folder, file_name);
    mask = imread(file_path);
    mask = bwlabel(mask, 4);
    adipo_ids = unique(mask);
    count_adipocytes = size(adipo_ids, 1) - 1;

    tokens = regexp(file_name, '^(.*)_([^_]*)$', 'tokens');
    if ~isempty(tokens)
        patient_ID = tokens{1}{1};
    else
        % If no underscores are found, use the whole file name
        [~, name, ~] = fileparts(file_name);
        patient_ID = name;
    end

    slide_name = [patient_ID '.svs'];

    T_temp = table({file_name}, {patient_ID}, {slide_name}, count_adipocytes);
    T = [T; T_temp];
end
repeatedStrings = repmat({dataset_name}, i, 1);
T.Properties.VariableNames{1} = 'file_name';
T.Properties.VariableNames{2} = 'patient_ID';
T.Properties.VariableNames{3} = 'slide_name';
T.Properties.VariableNames{4} = 'count_adipocytes';
T = [T repeatedStrings];
T.Properties.VariableNames{5} = 'dataset';
writetable(T, fullfile(output_path, [dataset_name '_dataset_info.xlsx']));