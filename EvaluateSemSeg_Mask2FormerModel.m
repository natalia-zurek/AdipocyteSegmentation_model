% to evaluate semantic segmentation use annotation folder as gnd truth
pred_datastore_path = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\GTEX 1024\mat';
gnd_datastore_path = 'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations unet GTEX 1024';
save_pth = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\evaluation\semantic seg';
name = 'Ov1 MTC aug 1024_GTEX';

dsPred = fileDatastore(pred_datastore_path, ...
    ReadFcn=@(x)predDataReaderSemseg(x));

dsGND = fileDatastore(gnd_datastore_path, ...
    ReadFcn=@(x)gndDataReaderSemseg(x));

classNames = ["background", "adipocyte"];
labelIDs = [0, 1];

tic
metrics = evaluateSemanticSegmentation(dsPred,dsGND);
t = toc/60
ds_metric = [metrics.DataSetMetrics; metrics.DataSetMetrics];
ds_metric.Properties.VariableNames{5} = 'MeanBFScoreDS';
T = [ds_metric, metrics.NormalizedConfusionMatrix, metrics.ClassMetrics];
writetable(T, fullfile(save_pth, [name '.xlsx']), "WriteRowNames",true);
save(fullfile(save_pth, [name '.mat']),'metrics')
%%
test = predDataReaderSemseg_postproc(dsPred2.Files{1, 1});
% test2 = gndDataReaderSemseg(dsGND.Files{256, 1});
%% COMBINED DATASET
save_pth = 'C:\_research_projects\Adipocyte model project\Mask2Former_v1\evaluation\model Ov1 MTC aug 1024\semantic seg';

pred_datastore_paths = {'C:\_research_projects\Adipocyte model project\Mask2Former_v1\predictions\model Ov1 MTC aug 1024\images GTEX 1024 normal infer\mat';...
    'C:\_research_projects\Adipocyte model project\Mask2Former_v1\predictions\model Ov1 MTC aug 1024\images TCGA 1024 normal infer\mat';...
    'C:\_research_projects\Adipocyte model project\Mask2Former_v1\predictions\model Ov1 MTC aug 1024\student project 1024 normal infer\mat'};

gnd_datastore_paths = {'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations unet GTEX 1024';...
    'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations TCGA 1024';...
    'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations student project 1024'};


dsPred2 = fileDatastore(pred_datastore_paths, ...
    ReadFcn=@(x)predDataReaderSemseg(x), FileExtensions='.mat');

dsGND2 = fileDatastore(gnd_datastore_paths, ...
    ReadFcn=@(x)gndDataReaderSemseg(x), FileExtensions='.mat');

tic
metrics = evaluateSemanticSegmentation(dsPred2,dsGND2, "Verbose",true);
t = toc/60

name = 'Ov1 MTC aug 1024_SP_TCGA_GTEX_combined';
ds_metric = [metrics.DataSetMetrics; metrics.DataSetMetrics];
ds_metric.Properties.VariableNames{5} = 'MeanBFScoreDS';
T = [ds_metric, metrics.NormalizedConfusionMatrix, metrics.ClassMetrics];
writetable(T, fullfile(save_pth, [name '.xlsx']), "WriteRowNames",true);
save(fullfile(save_pth, [name '.mat']),'metrics')
%% postprocessed data
pred_datastore_path = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\student project 1024\mat';
gnd_datastore_path = 'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations student project 1024';
save_pth = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\evaluation\semantic seg';
name = 'Ov1 MTC aug 1024_student project';

dsPred = fileDatastore(pred_datastore_path, ...
    ReadFcn=@(x)predDataReaderSemseg_postproc(x));

dsGND = fileDatastore(gnd_datastore_path, ...
    ReadFcn=@(x)gndDataReaderSemseg(x));

classNames = ["background", "adipocyte"];
labelIDs = [0, 1];

tic
metrics = evaluateSemanticSegmentation(dsPred,dsGND);
t = toc/60
ds_metric = [metrics.DataSetMetrics; metrics.DataSetMetrics];
ds_metric.Properties.VariableNames{5} = 'MeanBFScoreDS';
T = [ds_metric, metrics.NormalizedConfusionMatrix, metrics.ClassMetrics];
writetable(T, fullfile(save_pth, [name '.xlsx']), "WriteRowNames",true);
save(fullfile(save_pth, [name '.mat']),'metrics')
%% Combined datasets postprocess
save_pth = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\evaluation\semantic seg\';

pred_datastore_paths = {'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\GTEX 1024\mat';...
    'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\TCGA 1024\mat';...
    'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\student project 1024\mat'};

gnd_datastore_paths = {'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations unet GTEX 1024';...
    'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations TCGA 1024';...
    'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations student project 1024'};

dsPred2 = fileDatastore(pred_datastore_paths, ...
    ReadFcn=@(x)predDataReaderSemseg_postproc(x), FileExtensions='.mat');

dsGND2 = fileDatastore(gnd_datastore_paths, ...
    ReadFcn=@(x)gndDataReaderSemseg(x), FileExtensions='.mat');

tic
metrics = evaluateSemanticSegmentation(dsPred2,dsGND2, "Verbose",true);
t = toc/60

name = 'Mask2Former_Ov1_MTC_aug_1024_combined';
ds_metric = [metrics.DataSetMetrics; metrics.DataSetMetrics];
ds_metric.Properties.VariableNames{5} = 'MeanBFScoreDS';
T = [ds_metric, metrics.NormalizedConfusionMatrix, metrics.ClassMetrics];
writetable(T, fullfile(save_pth, [name '.xlsx']), "WriteRowNames",true);
save(fullfile(save_pth, [name '.mat']),'metrics')
%%

function out = predDataReaderSemseg_postproc(data_path)
classNames = ["background", "adipocyte"];
labelIDs = [0, 1];
load(data_path)

if isempty(inst_ids)
out = categorical(inst_map,labelIDs,classNames);
else
inst_map(inst_map > 0) = 1;
out = categorical(inst_map,labelIDs,classNames);

end

end


function out = predDataReaderSemseg(data_path)
classNames = ["background", "adipocyte"];
labelIDs = [0, 1];
load(data_path)

if isempty(inst_ids)
inst_map(inst_map == -1) = 0;
out = categorical(inst_map,labelIDs,classNames);
else
inst_map(inst_map == 0) = inst_ids(end)+1;
inst_map(inst_map == -1) = 0;
inst_map(inst_map > 0) = 1;
out = categorical(inst_map,labelIDs,classNames);

end

end

function out = gndDataReaderSemseg(data_path)
classNames = ["background", "adipocyte"];
labelIDs = [0, 1];
load(data_path)
out = categorical(class_map,labelIDs,classNames);

end

