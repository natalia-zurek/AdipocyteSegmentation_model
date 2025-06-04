%% ------ Mask2Former evaluation -------
clear all; clc; close all;
%% ------ EVALUATE SEMANTIC SEGMENTATION -------
pred_datastore_path = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former post\OM2 1024';
gnd_datastore_path = 'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations OM2';
save_pth = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former post\evaluation\instance seg';
out_name = 'Mask2Former_Ov1_MTC_aug_1024_OM2';

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

%% ------ EVALUATE INSTANCE SEGMENTATION ------
pred_datastore_path = '';
gnd_datastore_path = '';
save_pth = '';
out_name2 = '';
mkdir(save_pth);

dsPred = fileDatastore(pred_datastore_path, ...
    ReadFcn=@(x)predDataReaderInstseg(x), FileExtensions='.mat');

dsGND = fileDatastore(gnd_datastore_path, ...
    ReadFcn=@(x)gndDataReaderInstseg(x), FileExtensions='.mat');
tic
metrics = evaluateInstanceSegmentation(dsPred,dsGND, 0.5, "Verbose",true);
t = toc/60

save(fullfile(save_pth, [out_name2 '.mat']),'metrics')

%% ROC Curve
precision = metrics.ClassMetrics.Precision{1,1}(1:end);
recall = metrics.ClassMetrics.Recall{1,1}(1:end);

[sorted_recall, sortOrder] = sort(recall);
sorted_precision = precision(sortOrder);

f = figure(1);
plot(sorted_recall, sorted_precision)
out_name = 'Ov1 MTC aug 1024 - combined datasets';
title(out_name)
ylabel("Recall")
xlabel("Precision")

f.Position = [0 0 800 500];
exportgraphics(f,fullfile(save_pth, [out_name '_ROC.tif']))
%% DataReader functions for instance segmentation

% % % function out = predDataReader_postproc(data_path)
% % % load(data_path)
% % % 
% % % if isempty(inst_ids)
% % %     out{1} = logical(inst_map);
% % %     out{2} = categorical(repmat({'Adipocyte'}, 1, 1));
% % %     out{3} = 1;
% % % else
% % %     [inst_ids, exists_in_map ] = clean_ids(inst_map, inst_ids);
% % % 
% % %     mask = instancemask2maskstack(inst_map);
% % %     out{1} = mask;
% % % 
% % %     N=size(inst_ids, 2);
% % %     class_vector = categorical(repmat({'Adipocyte'}, N, 1));
% % %     out{2} = class_vector;
% % % 
% % %     inst_scores(exists_in_map == 0) = [];
% % %     out{3} = inst_scores';
% % % end
% % % end

function out = predDataReaderInstseg(data_path)
load(data_path)

if isempty(inst_ids)
inst_map(inst_map == -1) = 0;
out{1} = logical(inst_map);
out{2} = categorical(repmat({'Adipocyte'}, 1, 1));
out{3} = 1;

else
inst_map(inst_map == 0) = inst_ids(end)+1;
inst_map(inst_map == -1) = 0;
inst_ids(inst_ids == 0) = inst_ids(end)+1;
[inst_ids, exists_in_map ] = clean_ids(inst_map, inst_ids);

mask = instancemask2maskstack(inst_map);
out{1} = mask;

N=size(inst_ids, 2);
class_vector = categorical(repmat({'Adipocyte'}, N, 1));
out{2} = class_vector;


inst_scores(exists_in_map == 0) = [];
out{3} = inst_scores';
end
end

function out = gndDataReaderInstseg(data_path)

load(data_path)
% class_map = imread(data_path);
% inst_map = bwlabel(class_map, 4);
inst_maskstack = instancemask2maskstack(inst_map);
out{1} = inst_maskstack;

ids = unique(inst_map);
ids(ids == 0) = [];
N=size(ids, 1);
class_vector = categorical(repmat({'Adipocyte'}, N, 1));
out{2} = class_vector;

end
%% DataReader functions for semantic segmentation
function out = gndDataReaderSemseg(data_path)
classNames = ["background", "adipocyte"];
labelIDs = [0, 1];
load(data_path)
out = categorical(class_map,labelIDs,classNames);

end

function out = predDataReaderSemseg(data_path)
classNames = ["background", "adipocyte"];
labelIDs = [0, 1];
load(data_path)

if isempty(inst_ids)
    inst_map(inst_map == -1) = 0;
    out = categorical(inst_map,labelIDs,classNames);
else
    ids = unique(inst_map);
    if isempty(find(ids == -1, 1))
        inst_map(inst_map > 0) = 1;
        out = categorical(inst_map,labelIDs,classNames);
    else

        inst_map(inst_map == 0) = inst_ids(end)+1;
        inst_map(inst_map == -1) = 0;
        inst_map(inst_map > 0) = 1;
        out = categorical(inst_map,labelIDs,classNames);
    end
end
end

function [inst_ids, exists_in_map] = clean_ids(inst_map, inst_ids)

map_ids = unique(inst_map);
map_ids(map_ids == 0) = [];

exists_in_map = ismember(inst_ids, int32(map_ids));

% Remove elements from vector B that are not in vector A
inst_ids(exists_in_map == 0) = [];

end
