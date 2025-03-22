main_pth = "C:\_research_projects\Adipocyte model project\Mask2Former_v1\training dataset\model Ov1 MTC 1024\";
mask_pth = fullfile(main_pth, "annotations");
% image_path = fullfile(main_pth, "images/images TCGA/");
output_path = 'C:\Users\wylezoln\Box\Ovarian Cancer Computational Pathology\Adipocyte analysis\manuscript\MANUSCRIPT ADIPO\Natalia working folder\_tables';
% mkdir(output_path);
dataset_name = 'OM1';
files = dir(fullfile(mask_pth, '*.mat'));

T = table();
for i = 1:size(files, 1)
    file_name = files(i).name;
    file_path = fullfile(files(i).folder, file_name);

    % mask = imread(file_path);
    % mask = bwlabel(mask, 4);

    load(file_path);
    mask = inst_map;

    adipo_ids = unique(mask);
    count_adipocytes = size(adipo_ids, 1) - 1;

    % % tokens = regexp(file_name, '^(.*)_([^_]*)$', 'tokens');
    % % if ~isempty(tokens)
    % %     patient_ID = tokens{1}{1};
    % % else
    % %     % If no underscores are found, use the whole file name
    % %     [~, name, ~] = fileparts(file_name);
    % %     patient_ID = name;
    % % end
    tokens = strfind(file_name, '_');
    if ~isempty(tokens)
        patient_ID = file_name(1:tokens(1)-1);
        if length(tokens) == 1
            aug = 0;
        else

            aug = 1;

        end
    else
        error('File name does not match the expected format: mainName_X_Y.mat or mainName_X.mat');
    end


    slide_name = [patient_ID '.svs'];

    T_temp = table({file_name}, {patient_ID}, {slide_name}, aug, count_adipocytes);
    T = [T; T_temp];
end
repeatedStrings = repmat({dataset_name}, i, 1);
T.Properties.VariableNames{1} = 'file_name';
T.Properties.VariableNames{2} = 'patient_ID';
T.Properties.VariableNames{3} = 'slide_name';
T.Properties.VariableNames{4} = 'augmented';
T.Properties.VariableNames{5} = 'count_adipocytes';
T = [T repeatedStrings];
T.Properties.VariableNames{6} = 'dataset';

writetable(T, fullfile(output_path, 'training_dataset_info.xlsx'), 'Sheet', dataset_name);