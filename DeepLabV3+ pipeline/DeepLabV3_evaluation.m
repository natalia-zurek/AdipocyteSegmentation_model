%% ------ EVALUATE SEMANTIC SEGMENTATION -------
pred_datastore_path = '';
gnd_datastore_path = '';
save_pth = '';

out_name = '';

dsPred = fileDatastore(pred_datastore_path, ...
    ReadFcn=@(x)predDataReaderSemseg(x), FileExtensions='.mat');

dsGND = fileDatastore(gnd_datastore_path, ...
    ReadFcn=@(x)gndDataReaderSemseg(x), FileExtensions='.mat');

classNames = ["background", "adipocyte"];
labelIDs = [0, 1];

tic
metrics = evaluateSemanticSegmentation(dsPred,dsGND);
t = toc/60
ds_metric = [metrics.DataSetMetrics; metrics.DataSetMetrics];
ds_metric.Properties.VariableNames{5} = 'MeanBFScoreDS';
T = [ds_metric, metrics.NormalizedConfusionMatrix, metrics.ClassMetrics];
writetable(T, fullfile(save_pth, [out_name '.xlsx']), "WriteRowNames",true);
save(fullfile(save_pth, [out_name '.mat']),'metrics')

%% Combined datasets
save_pth = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\DeepLabV3plus raw\evaluation\semantic seg';
out_name2 = 'DLV_raw_OM2_GTEX_TCGA';

pred_datastore_paths = {'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\DeepLabV3plus raw\OM2 1024';...
    'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\DeepLabV3plus raw\GTEX 1024';...
    'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\DeepLabV3plus raw\TCGA 1024'};
gnd_datastore_paths = {'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations OM2';...
    'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations unet GTEX 1024';...
    'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations TCGA 1024'};

dsPred = fileDatastore(pred_datastore_paths, ...
    ReadFcn=@(x)predDataReaderSemseg(x), FileExtensions='.mat');

dsGND = fileDatastore(gnd_datastore_paths, ...
    ReadFcn=@(x)gndDataReaderSemseg(x), FileExtensions='.mat');

tic
metrics = evaluateSemanticSegmentation(dsPred,dsGND, "Verbose",true);
t = toc/60

ds_metric = [metrics.DataSetMetrics; metrics.DataSetMetrics];
ds_metric.Properties.VariableNames{5} = 'MeanBFScoreDS';
T = [ds_metric, metrics.NormalizedConfusionMatrix, metrics.ClassMetrics];
writetable(T, fullfile(save_pth, [out_name2 '.xlsx']), "WriteRowNames",true);
T2 = table(dsPred.Files);
T2 = [T2, metrics.ImageMetrics];
writetable(T2, fullfile(save_pth, [out_name2 '.xlsx']), 'Sheet', 'image metrics')
save(fullfile(save_pth, [out_name2 '.mat']),'metrics')

%% ------ EVALUATE INSTANCE SEGMNTATION -------
pred_datastore_path = '';
gnd_datastore_path = '';
save_pth = '';
out_name = '';

mkdir(save_pth);

dsPred = fileDatastore(pred_datastore_path, ...
    ReadFcn=@(x)predDataReaderInstseg(x), FileExtensions='.mat');

dsGND = fileDatastore(gnd_datastore_path, ...
    ReadFcn=@(x)gndDataReaderInstseg(x), FileExtensions='.mat');
tic
metrics = evaluateInstanceSegmentation(dsPred, dsGND, 0.5, "Verbose", true);
t = toc/60


save(fullfile(save_pth, [out_name '.mat']),'metrics')

%% combined datasets
% pred_datastore_paths = {'';...
%     '';...
%     ''};
% gnd_datastore_paths = {'';...
%     '';...
%     ''};

pred_datastore_paths = {'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\DeepLabV3plus post\OM2 1024';...
    'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\DeepLabV3plus post\GTEX 1024\mat';...
    'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\DeepLabV3plus post\TCGA 1024\mat'};
gnd_datastore_paths = {'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations OM2';...
    'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations unet GTEX 1024';...
    'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations TCGA 1024'};

out_name2 = 'DeepLabV3plus_post_Ov1_MTC_aug_1024_OM2_GTEX_TCGA_v2';
save_pth = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\DeepLabV3plus post\evaluation\instance seg';
mkdir(save_pth);

dsPred = fileDatastore(pred_datastore_paths, ...
    ReadFcn=@(x)predDataReaderInstseg(x), FileExtensions='.mat');

dsGND = fileDatastore(gnd_datastore_paths, ...
    ReadFcn=@(x)gndDataReaderInstseg(x), FileExtensions='.mat');

tic
metrics = evaluateInstanceSegmentation(dsPred, dsGND, 0.5, "Verbose", true);
t = toc/60

save(fullfile(save_pth, [out_name2 '.mat']), 'metrics')


%% Datastore readers for instance segmentation
test2 = predDataReaderInstseg(dsPred.Files{1, 1});
%%
test_post = predDataReader_postproc(dsPred.Files{1, 1});
%%
function out = predDataReaderInstseg(data_path)

load(data_path)
% mask = bwlabel(mask, 8);
inst_ids = unique(mask);
inst_ids(inst_ids == 0) = [];

if isempty(inst_ids)
    out{1} = logical(mask);
    out{2} = categorical(repmat({'Adipocyte'}, 1, 1));
    out{3} = 1;

else

    mask = instancemask2maskstack(mask);
    out{1} = mask;

    N=size(inst_ids, 1);
    class_vector = categorical(repmat({'Adipocyte'}, N, 1));
    out{2} = class_vector;

    inst_scores = repmat(0.5, N, 1);
    % inst_scores = ones(N, 1);
    out{3} = inst_scores;

end
end

function out = predDataReader_postproc(data_path)

load(data_path)
inst_ids = unique(mask);
inst_ids(inst_ids == 0) = [];

if isempty(inst_ids)
out{1} = logical(mask);
out{2} = categorical(repmat({'Adipocyte'}, 1, 1));
out{3} = 1;

else

mask = instancemask2maskstack(mask);
out{1} = mask;

N=size(inst_ids, 1);
class_vector = categorical(repmat({'Adipocyte'}, N, 1));
out{2} = class_vector;

inst_scores = repmat(0.5, N, 1);
out{3} = inst_scores;

end
end

function out = gndDataReaderInstseg(data_path)

load(data_path)
inst_maskstack = instancemask2maskstack(inst_map);
out{1} = inst_maskstack;

ids = unique(inst_map);
ids(ids == 0) = [];
N=size(ids, 1);
class_vector = categorical(repmat({'Adipocyte'}, N, 1));
out{2} = class_vector;

end
%% DataReader functions for semantic segmentation
function out = predDataReaderSemseg(data_path)
classNames = ["background", "adipocyte"];
labelIDs = [0, 1];
load(data_path)
mask(mask > 0) = 1; %ensure binary
out = categorical(mask,labelIDs,classNames);
end

function out = gndDataReaderSemseg(data_path)
classNames = ["background", "adipocyte"];
labelIDs = [0, 1];
load(data_path)
class_map(class_map > 0) = 1; %ensure binary
out = categorical(class_map,labelIDs,classNames);
end
