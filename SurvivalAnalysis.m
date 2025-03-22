%% OVARIAN CANCER
addpath(genpath('C:\_research_projects\Immune infiltrate project\immune infiltrate\Immune-infiltrate-project'));
addpath(genpath('C:\_research_projects\Research-scripts'))
data_M2Fraw = readtable('C:/_research_projects/Adipocyte model project/Adipocyte analysis/Omental Mets/Mask2Former/Mask2Former raw/adipocyte_morphological_features_OM_aggregated_Mask2Former.xlsx',VariableNamingRule="preserve");
data_M2Fpost = readtable('C:/_research_projects/Adipocyte model project/Adipocyte analysis/Omental Mets/Mask2Former/Mask2Former postprocessed/adipocyte_morphological_features_OM_aggregated_Mask2Former_postproc.xlsx', VariableNamingRule="preserve");

data_DLVraw = readtable('C:/_research_projects/Adipocyte model project/Adipocyte analysis/Omental Mets/DeepLabV3plus/DeepLabV3plus raw/adipocyte_morphological_features_OM_aggregated_DeepLabV3plus.xlsx', VariableNamingRule="preserve");
data_DLVpost = readtable('C:/_research_projects/Adipocyte model project/Adipocyte analysis/Omental Mets/DeepLabV3plus/DeepLabV3plus postprocessed/adipocyte_morphological_features_OM_aggregated_DeepLabV3plus_postproc.xlsx', VariableNamingRule="preserve");
%%
clinical_data = readtable('C:/_research_projects/Adipocyte model project/Adipocyte analysis/Omental Mets/Omental_clinical_data.csv', VariableNamingRule="preserve");
clinical_data(clinical_data.Patient_Deidentified_ID == 896, :) = [];
clinical_data(clinical_data.Patient_Deidentified_ID == 12245, :) = [];
clinical_data(clinical_data.Patient_Deidentified_ID == 12261, :) = [];
clinical_data.Patient_Deidentified_ID = strcat(num2str(clinical_data.Patient_Deidentified_ID), '.svs');
clinical_data.Patient_Deidentified_ID = cellstr(clinical_data.Patient_Deidentified_ID);
clinical_data.Properties.VariableNames(4) = "Slide name";
%%
data = data_DLVpost;
name = 'DLVpost';

KM_save = fullfile('C:\_research_projects\Adipocyte model project\Adipocyte analysis\Omental Mets\Survival analysis\KM_median\', name);
KM_save2 = fullfile('C:\_research_projects\Adipocyte model project\Adipocyte analysis\Omental Mets\Survival analysis\KM_best_quartile\', name);
filename_excel = fullfile(KM_save, ['Surv_KM_median_' name '.xlsx']);
filename_excel2 = fullfile(KM_save2, ['Surv_KM_best_qt_' name '.xlsx']);


mkdir(KM_save);
mkdir(KM_save2);
cut_point = 'Median'; % 'Median' 'Quartile', QuartileAll or 'Tertile' %quartile = 25, 75, quartileAll = 25, 50, 75, tertile = [100/3 100/1.5]
T = innerjoin(clinical_data, data, "Keys","Slide name");
%%
data_M2Fraw = readtable(fullfile(pth, 'Adipocyte_ft_M2Fraw_Omental_mets_clinical.xlsx'), 'VariableNamingRule','preserve'); 
data_M2Fpost = readtable(fullfile(pth, 'Adipocyte_ft_M2Fpost_Omental_mets_clinical.xlsx'), 'VariableNamingRule','preserve'); 

data_DLVraw = readtable(fullfile(pth, 'Adipocyte_ft_DLVraw_Omental_mets_clinical.xlsx'), 'VariableNamingRule','preserve'); 
data_DLVpost = readtable(fullfile(pth, 'Adipocyte_ft_DLVpost_Omental_mets_clinical.xlsx'), 'VariableNamingRule','preserve'); 
%%
data = data_M2Fpost;
name = 'M2Fpost';

KM_save = fullfile('C:\_research_projects\Adipocyte model project\Adipocyte analysis\Omental Mets\Survival analysis_v2\KM_median\', name);
KM_save2 = fullfile('C:\_research_projects\Adipocyte model project\Adipocyte analysis\Omental Mets\Survival analysis_v2\KM_best_quartile\', name);
filename_excel = fullfile(KM_save, ['Surv_KM_median_' name '.xlsx']);
filename_excel2 = fullfile(KM_save2, ['Surv_KM_best_qt_' name '.xlsx']);


mkdir(KM_save);
mkdir(KM_save2);
cut_point = 'Median'; % 'Median' 'Quartile', QuartileAll or 'Tertile' %quartile = 25, 75, quartileAll = 25, 50, 75, tertile = [100/3 100/1.5]

preds = data.Properties.VariableNames(37:end)';
trh = 0.001;  %removes invariant features
[data, preds, removedFs] = QCNfiler2(data, preds, trh);
time =  'Overall.Survival..Time.to.Death.or.to.last.survival.status.if_1';
event = 'Patient.status..1.dead..0.alive.';

kmplotter_workflow(data, preds, time, event, filename_excel, KM_save, cut_point)
kmplotter_workflow_best_quartile(data, preds, time, event, filename_excel2, KM_save2)

%% PLCO
addpath(genpath('C:\_research_projects\Immune infiltrate project\immune infiltrate\Immune-infiltrate-project'));
addpath(genpath('C:\_research_projects\Research-scripts'))
data_M2Fraw = readtable('C:\_research_projects\Adipocyte model project\Adipocyte analysis\PLCO\Mask2Former\Mask2Former raw/adipocyte_morphological_features_PLCO_aggregated_Mask2Former_raw.xlsx',VariableNamingRule="preserve");
data_M2Fpost = readtable('C:/_research_projects/Adipocyte model project/Adipocyte analysis/PLCO/Mask2Former/Mask2Former postproc/adipocyte_morphological_features_PLCO_aggregated_Mask2Former_postproc.xlsx', VariableNamingRule="preserve");

data_DLVraw = readtable('C:\_research_projects\Adipocyte model project\Adipocyte analysis\PLCO\DeepLabV3plus\DeepLabV3plus raw/adipocyte_morphological_features_PLCO_aggregated_DeepLabV3plus_raw', VariableNamingRule="preserve");
data_DLVpost = readtable('C:\_research_projects\Adipocyte model project\Adipocyte analysis\PLCO\DeepLabV3plus\DeepLabV3plus postproc/adipocyte_morphological_features_PLCO_aggregated_DeepLabV3plus_postproc.xlsx', VariableNamingRule="preserve");

clinical_data = readtable('C:/_research_projects/Adipocyte model project/Adipocyte analysis/PLCO/PLCO_clinical_data.csv', VariableNamingRule="preserve");
clinical_data.Properties.VariableNames(3) = "Patient_ID";
%%
data = data_M2Fpost;
name = 'M2Fpost';

KM_save = fullfile('C:\_research_projects\Adipocyte model project\Adipocyte analysis\PLCO\Survival analysis\KM_median\', name);
KM_save2 = fullfile('C:\_research_projects\Adipocyte model project\Adipocyte analysis\PLCO\Survival analysis\KM_best_quartile\', name);
filename_excel = fullfile(KM_save, ['Surv_KM_median_' name '.xlsx']);
filename_excel2 = fullfile(KM_save2, ['Surv_KM_best_qt_' name '.xlsx']);

mkdir(KM_save);
mkdir(KM_save2);
cut_point = 'Median'; % 'Median' 'Quartile', QuartileAll or 'Tertile' %quartile = 25, 75, quartileAll = 25, 50, 75, tertile = [100/3 100/1.5]
%%
T = innerjoin(clinical_data(:, 3:end), data, "Keys","Patient_ID");
[~,ia] = unique(T.Patient_ID);
T = T(ia,:);

preds = T.Properties.VariableNames(110:end)';
trh = 0.001;  %removes invariant features
[data, preds, removedFs] = QCNfiler2(data, preds, trh);
time =  'Overall_Survival_calculated';
event = 'is_dead';

kmplotter_workflow(T, preds, time, event, filename_excel, KM_save, cut_point)
kmplotter_workflow_best_quartile(T, preds, time, event, filename_excel2, KM_save2)

%%
pth = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\PLCO\Early survival';
dlpost = readtable(fullfile(pth,"DLVpost_vs_clinical_pearson_correlation_p.xlsx"), 'VariableNamingRule','preserve');
dlraw = readtable(fullfile(pth,"DLVraw_vs_clinical_pearson_correlation_p.xlsx"), 'VariableNamingRule','preserve');
m2fpost = readtable(fullfile(pth,"M2Fpost_vs_clinical_pearson_correlation_p.xlsx"), 'VariableNamingRule','preserve');
m2fraw = readtable(fullfile(pth,"M2Fraw_vs_clinical_pearson_correlation_p.xlsx"), 'VariableNamingRule','preserve');
%%
pth = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\PLCO\Early survival';
dlpost = readtable(fullfile(pth,"DLVpost_early_survival_t_test_tstat.xlsx"), 'VariableNamingRule','preserve');
dlraw = readtable(fullfile(pth,"DLVraw_early_survival_t_test_tstat.xlsx"), 'VariableNamingRule','preserve');
m2fpost = readtable(fullfile(pth,"M2Fpost_early_survival_t_test_tstat.xlsx"), 'VariableNamingRule','preserve');
m2fraw = readtable(fullfile(pth,"M2Fraw_early_survival_t_test_tstat.xlsx"), 'VariableNamingRule','preserve');
%%
Tjoin = outerjoin(m2fraw, m2fpost, "Keys","names.feature2.", "MergeKeys",true);
T2 = outerjoin(Tjoin, dlraw, "Keys","names.feature2.", "MergeKeys",true);
T3 = outerjoin(T2, dlpost, "Keys","names.feature2.", "MergeKeys",true);
T3 = removevars(T3, "Var1_dlpost");
T3 = removevars(T3, "Var1_T2");
T3 = removevars(T3, "Var1_m2fraw");
T3 = removevars(T3, "Var1_m2fpost");
% Get variable names from each table
varNamesRaw = T3.Properties.VariableNames(2:8);
varNamesPost = T3.Properties.VariableNames(9:15);
varNamesDlRaw =T3.Properties.VariableNames(16:22);
varNamesDlPost = T3.Properties.VariableNames(23:29);

% Ensure all tables have the same number of columns
numCols = min([numel(varNamesRaw), numel(varNamesPost), numel(varNamesDlRaw), numel(varNamesDlPost)]);

% Create the interleaved column order
interleavedCols = {};
 interleavedCols = [interleavedCols, T3.Properties.VariableNames(1)];
for i = 1:numCols
    interleavedCols = [interleavedCols, varNamesRaw(i), varNamesPost(i), varNamesDlRaw(i), varNamesDlPost(i)];
end

% Apply the new column order
T3 = T3(:, interleavedCols);
%%

writetable(T3, fullfile(pth, "combined.xlsx"), "Sheet", 'p_val');




