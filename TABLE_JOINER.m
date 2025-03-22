pth = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Omental Mets\early survival analysis v2';
m2fraw = readtable(fullfile(pth, "M2Fraw_early_survival_t_test_tstat.xlsx"), 'VariableNamingRule','preserve');
m2fpost = readtable(fullfile(pth, "M2Fpost_early_survival_t_test_tstat.xlsx"), 'VariableNamingRule','preserve');
dlraw = readtable(fullfile(pth, "DLVraw_early_survival_t_test_tstat.xlsx"), 'VariableNamingRule','preserve');
dlpost = readtable(fullfile(pth, "DLVpost_early_survival_t_test_tstat.xlsx"), 'VariableNamingRule','preserve');

m2fraw(:, "Var1") = [];
m2fpost(:, "Var1") = [];
dlraw(:, "Var1") = [];
dlpost(:, "Var1") = [];
%%
% Rename columns to avoid duplicates
m2fraw.Properties.VariableNames = strcat(m2fraw.Properties.VariableNames, "_M2Fraw");
m2fpost.Properties.VariableNames = strcat(m2fpost.Properties.VariableNames, "_M2Fpost");
dlraw.Properties.VariableNames = strcat(dlraw.Properties.VariableNames, "_DLVraw");
dlpost.Properties.VariableNames = strcat(dlpost.Properties.VariableNames, "_DLVpost");

% Get original column names (before renaming)
colNames = erase(m2fraw.Properties.VariableNames, "_M2Fraw");

% Initialize a new table to store interleaved columns
combinedTable = table();

% Interleave columns from all four tables
for i = 1:numel(colNames)
    combinedTable = [combinedTable, m2fraw(:, i), m2fpost(:, i), dlraw(:, i), dlpost(:, i)];
end

% Save the combined table
writetable(combinedTable, fullfile(pth, "combined.xlsx"), "Sheet", 't-stat');
%%
pth = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Omental Mets\Survival analysis_v2\KM_median';
m2fraw = readtable(fullfile(pth, "M2Fraw/Surv_KM_median_M2Fraw.xlsx"), 'VariableNamingRule','preserve');
m2fpost = readtable(fullfile(pth, "M2Fpost/Surv_KM_median_M2Fpost.xlsx"), 'VariableNamingRule','preserve');
dlraw = readtable(fullfile(pth, "DLVraw/Surv_KM_median_DLVraw.xlsx"), 'VariableNamingRule','preserve');
dlpost = readtable(fullfile(pth, "DLVpost/Surv_KM_median_DLVpost.xlsx"), 'VariableNamingRule','preserve');

%%
% Outer join the tables based on column names (merge all data)
Tjoin = outerjoin(m2fraw, m2fpost, 'Keys','preds' ,'MergeKeys', true);
%%
Tjoin = outerjoin(Tjoin, dlraw, 'Keys','preds' ,'MergeKeys', true);
Tjoin = outerjoin(Tjoin, dlpost, 'Keys','preds' ,'MergeKeys', true);
%%

writetable(Tjoin, fullfile(pth, "combined.xlsx"), "Sheet", 'pval');

%%
folder_path = 'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations TCGA 1024';
img_path = 'C:\_research_projects\Adipocyte model project\Original data\images\images TCGA 1024';
output_path = 'C:\_research_projects\Adipocyte model project\Original data\overlay\overlay TCGA 1024';
mkdir(output_path);
files = dir(fullfile(folder_path, '*.mat'));
%%
for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);
    I = imread(fullfile(img_path, [name '.tif']));
    load(file_path)
    % mask = bwlabel(mask);
    % [inst_map, ~, ~, ~] = python2matlab_instseg(inst_map);
    ov = labeloverlay(I, inst_map, "Transparency", 0.5);

    imwrite(ov, fullfile(output_path, [name '.png']));
end