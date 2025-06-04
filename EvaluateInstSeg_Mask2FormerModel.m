pred_datastore_path = {'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former raw\images OM2 normal infer\mat',...
    'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former raw\images GTEX 1024 normal infer\mat', ...
    'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former raw\images TCGA 1024 normal infer\mat'};
gnd_datastore_path = {'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations OM2';...
    'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations unet GTEX 1024';...
    'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations TCGA 1024'};

% pred_datastore_path = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former raw\images OM2 normal infer\mat';
% gnd_datastore_path = 'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations OM2';
save_pth = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former raw\evaluation\instance seg';
out_name = 'Mask2Former_raw_Ov1_MTC_aug_1024_OM2_GTEX_TCGA';

dsPred = fileDatastore(pred_datastore_path, ...
    ReadFcn=@(x)predDataReaderInstseg(x), FileExtensions='.mat');

dsGND = fileDatastore(gnd_datastore_path, ...
    ReadFcn=@(x)gndDataReaderInstseg(x), FileExtensions='.mat');
tic
metrics = evaluateInstanceSegmentation(dsPred, dsGND, 0.5, "Verbose", true);
t = toc/60


save(fullfile(save_pth, [out_name '.mat']),'metrics')

%% DataReader functions for semantic segmentation
om2= predDataReaderInstseg(dsPred.Files{1, 1});
gtex = predDataReaderInstseg(dsPred.Files{123, 1});
tcga= predDataReaderInstseg(dsPred.Files{530, 1});
%%
function out = predDataReaderInstseg(data_path)
load(data_path)

if isempty(inst_ids)
inst_map(inst_map == -1) = 0;
out{1} = logical(inst_map);
out{2} = categorical(repmat({'Adipocyte'}, 1, 1));
out{3} = 1;

else
    ids = unique(inst_map);
    if ~isempty(find(ids == -1, 1))
        inst_map(inst_map == 0) = inst_ids(end)+1;
        inst_map(inst_map == -1) = 0;
        inst_ids(inst_ids == 0) = inst_ids(end)+1;
    end

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


% %% evaluate instance segmentation
% pred_datastore_path = 'C:\_research_projects\Adipocyte model project\Mask2Former_v1\predictions\model Ov1 MTC aug 1024\images GTEX 1024 normal infer\mat';
% gnd_datastore_path = 'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations unet GTEX 1024';
% dsPred = fileDatastore(pred_datastore_path, ...
%     ReadFcn=@(x)predDataReader(x));
% 
% dsGND = fileDatastore(gnd_datastore_path, ...
%     ReadFcn=@(x)gndDataReader2(x));
% tic
% metrics = evaluateInstanceSegmentation(dsPred,dsGND, 0.5, "Verbose",true);
% t = toc/60
% 
% save_pth = 'C:\_research_projects\Adipocyte model project\Mask2Former_v1\evaluation\model Ov1 MTC aug 1024 intratumoral fat/instance seg';
% mkdir(save_pth);
% save(fullfile(save_pth, 'Ov1 MTC aug 1024_GTEX.mat'),'metrics')
% 
% pred_datastore_path = 'C:\_research_projects\Adipocyte model project\Mask2Former_v1\predictions\model Ov1 MTC aug 1024 intratumoral fat\images TCGA 1024 normal infer\mat';
% gnd_datastore_path = 'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations TCGA 1024';
% dsPred = fileDatastore(pred_datastore_path, ...
%     ReadFcn=@(x)predDataReader(x));
% 
% dsGND = fileDatastore(gnd_datastore_path, ...
%     ReadFcn=@(x)gndDataReader2(x));
% tic
% metrics = evaluateInstanceSegmentation(dsPred,dsGND, 0.5, "Verbose",true);
% t = toc/60
% 
% save(fullfile(save_pth, 'Ov1 MTC aug 1024_TCGA.mat'),'metrics')
% 
% pred_datastore_path = 'C:\_research_projects\Adipocyte model project\Mask2Former_v1\predictions\model Ov1 MTC aug 1024 intratumoral fat\images student project 1024 normal infer\mat';
% gnd_datastore_path = 'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations student project 1024';
% dsPred = fileDatastore(pred_datastore_path, ...
%     ReadFcn=@(x)predDataReader(x));
% 
% dsGND = fileDatastore(gnd_datastore_path, ...
%     ReadFcn=@(x)gndDataReader2(x));
% tic
% metrics = evaluateInstanceSegmentation(dsPred,dsGND, 0.5, "Verbose",true);
% t = toc/60
% 
% save(fullfile(save_pth, 'Ov1 MTC aug 1024_student_project_1024.mat'),'metrics')
% %% ROC Curve
% precision = metrics.ClassMetrics.Precision{1,1}(1:end);
% recall = metrics.ClassMetrics.Recall{1,1}(1:end);
% 
% [sorted_recall, sortOrder] = sort(recall);
% sorted_precision = precision(sortOrder);
% 
% f = figure(1);
% plot(sorted_recall, sorted_precision)
% name = 'Ov1 MTC aug 1024 - combined datasets';
% title(name)
% ylabel("Recall")
% xlabel("Precision")
% 
% f.Position = [0 0 800 500];
% exportgraphics(f,fullfile(save_pth, [name '.tif']))
% %% COMBINED DATASET
% save_pth = 'C:\_research_projects\Adipocyte model project\Mask2Former_v1\evaluation\model Ov1 MTC aug 1024\instance seg';
% 
% pred_datastore_paths = {'C:\_research_projects\Adipocyte model project\Mask2Former_v1\predictions\model Ov1 MTC aug 1024\images GTEX 1024 normal infer\mat';...
%     'C:\_research_projects\Adipocyte model project\Mask2Former_v1\predictions\model Ov1 MTC aug 1024\images TCGA 1024 normal infer\mat';...
%     'C:\_research_projects\Adipocyte model project\Mask2Former_v1\predictions\model Ov1 MTC aug 1024\student project 1024 normal infer\mat'};
% 
% gnd_datastore_paths = {'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations unet GTEX 1024';...
%     'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations TCGA 1024';...
%     'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations student project 1024'};
% 
% 
% dsPred = fileDatastore(pred_datastore_paths, ...
%     ReadFcn=@(x)predDataReader(x), FileExtensions='.mat');
% 
% dsGND = fileDatastore(gnd_datastore_paths, ...
%     ReadFcn=@(x)gndDataReader2(x), FileExtensions='.mat');
% tic
% metrics = evaluateInstanceSegmentation(dsPred,dsGND, 0.5, "Verbose",true);
% t = toc/60
% 
% save(fullfile(save_pth, 'Ov1 MTC aug 1024_SP_TCGA_GTEX_combined.mat'),'metrics')
% %% evaluate postprocessed
% save_pth = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed/evaluation/instance seg';
% mkdir(save_pth);
% 
% pred_datastore_path = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\GTEX 1024\mat';
% gnd_datastore_path = 'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations unet GTEX 1024';
% dsPred = fileDatastore(pred_datastore_path, ...
%     ReadFcn=@(x)predDataReader_postproc(x));
% 
% dsGND = fileDatastore(gnd_datastore_path, ...
%     ReadFcn=@(x)gndDataReader2(x));
% tic
% metrics = evaluateInstanceSegmentation(dsPred,dsGND, 0.5, "Verbose",true);
% t = toc/60
% 
% 
% save(fullfile(save_pth, 'Ov1 MTC aug 1024_GTEX.mat'),'metrics')
% 
% pred_datastore_path = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\TCGA 1024\mat';
% gnd_datastore_path = 'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations TCGA 1024';
% dsPred = fileDatastore(pred_datastore_path, ...
%     ReadFcn=@(x)predDataReader_postproc(x));
% 
% dsGND = fileDatastore(gnd_datastore_path, ...
%     ReadFcn=@(x)gndDataReader2(x));
% tic
% metrics = evaluateInstanceSegmentation(dsPred,dsGND, 0.5, "Verbose",true);
% t = toc/60
% 
% save(fullfile(save_pth, 'Ov1 MTC aug 1024_TCGA.mat'),'metrics')
% 
% pred_datastore_path = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\student project 1024\mat';
% gnd_datastore_path = 'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations student project 1024';
% dsPred = fileDatastore(pred_datastore_path, ...
%     ReadFcn=@(x)predDataReader_postproc(x));
% 
% dsGND = fileDatastore(gnd_datastore_path, ...
%     ReadFcn=@(x)gndDataReader2(x));
% tic
% metrics = evaluateInstanceSegmentation(dsPred,dsGND, 0.5, "Verbose",true);
% t = toc/60
% 
% save(fullfile(save_pth, 'Ov1 MTC aug 1024_student_project_1024.mat'),'metrics')
% %% combined postprocessed
% save_pth = 'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\evaluation\instance seg';
% pred_datastore_paths = {'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\GTEX 1024\mat';...
%     'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\TCGA 1024\mat';...
%     'C:\_research_projects\Adipocyte model project\Adipocyte analysis\Test datasets\Mask2Former postprocessed\student project 1024\mat'};
% 
% gnd_datastore_paths = {'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations unet GTEX 1024';...
%     'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations TCGA 1024';...
%     'C:\_research_projects\Adipocyte model project\Original data\annotations\annotations student project 1024'};
% 
% dsPred = fileDatastore(pred_datastore_paths, ...
%     ReadFcn=@(x)predDataReader_postproc(x));
% 
% dsGND = fileDatastore(gnd_datastore_paths, ...
%     ReadFcn=@(x)gndDataReader2(x));
% tic
% metrics = evaluateInstanceSegmentation(dsPred,dsGND, 0.5, "Verbose",true);
% t = toc/60
% save(fullfile(save_pth, 'Ov1 MTC aug 1024_SP_TCGA_GTEX_combined.mat'),'metrics')
% %%
% test = predDataReader_postproc(dsPred.Files{1, 1});
% %%
% function out = predDataReader_postproc(data_path)
% 
% load(data_path)
% 
% if isempty(inst_ids)
%     out{1} = logical(inst_map);
%     out{2} = categorical(repmat({'Adipocyte'}, 1, 1));
%     out{3} = 1;
% else
%     [inst_ids, exists_in_map ] = clean_ids(inst_map, inst_ids);
% 
%     mask = instancemask2maskstack(inst_map);
%     out{1} = mask;
% 
%     N=size(inst_ids, 2);
%     class_vector = categorical(repmat({'Adipocyte'}, N, 1));
%     out{2} = class_vector;
% 
%     inst_scores(exists_in_map == 0) = [];
%     out{3} = inst_scores';
% end
% end
% 
% 
% function out = predDataReader(data_path)
% 
% load(data_path)
% 
% if isempty(inst_ids)
% inst_map(inst_map == -1) = 0;
% out{1} = logical(inst_map);
% out{2} = categorical(repmat({'Adipocyte'}, 1, 1));
% out{3} = 1;
% 
% else
% inst_map(inst_map == 0) = inst_ids(end)+1;
% inst_map(inst_map == -1) = 0;
% inst_ids(inst_ids == 0) = inst_ids(end)+1;
% [inst_ids, exists_in_map ] = clean_ids(inst_map, inst_ids);
% 
% mask = instancemask2maskstack(inst_map);
% out{1} = mask;
% 
% N=size(inst_ids, 2);
% class_vector = categorical(repmat({'Adipocyte'}, N, 1));
% out{2} = class_vector;
% 
% 
% inst_scores(exists_in_map == 0) = [];
% out{3} = inst_scores';
% end
% end
% 
% function out = gndDataReader2(data_path)
% 
% load(data_path)
% % class_map = imread(data_path);
% % inst_map = bwlabel(class_map, 4);
% inst_maskstack = instancemask2maskstack(inst_map);
% out{1} = inst_maskstack;
% 
% ids = unique(inst_map);
% ids(ids == 0) = [];
% N=size(ids, 1);
% class_vector = categorical(repmat({'Adipocyte'}, N, 1));
% out{2} = class_vector;
% 
% end
% 
% function out = gndDataReader(data_path)
% 
% class_map = imread(data_path);
% inst_map = bwlabel(class_map, 4);
% inst_maskstack = instancemask2maskstack(inst_map);
% out{1} = inst_maskstack;
% 
% ids = unique(inst_map);
% ids(ids == 0) = [];
% N=size(ids, 1);
% class_vector = categorical(repmat({'Adipocyte'}, N, 1));
% out{2} = class_vector;
% 
% end
% 
function [inst_ids, exists_in_map ]= clean_ids(inst_map, inst_ids)

map_ids = unique(inst_map);
map_ids(map_ids == 0) = [];

exists_in_map = ismember(inst_ids, int32(map_ids));

% Remove elements from vector B that are not in vector A
inst_ids(exists_in_map == 0) = [];

end
