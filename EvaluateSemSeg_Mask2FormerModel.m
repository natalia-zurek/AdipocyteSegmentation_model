pred_datastore_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\predictions\model Ov1 MTC aug 512 no_glands\abdominal laparoscopy 10x\masks';
%gnd_datastore_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\test dataset\omental mets part 2\masks';
gnd_datastore_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\test dataset\abdominal_laparoscopy\masks 10x';

output_path = fullfile(pred_datastore_path, 'binary mask');
mkdir(output_path);
files = [dir(fullfile(pred_datastore_path, '*.jpg')); dir(fullfile(pred_datastore_path, '*.png'))];
%%
for i = 1:size(files, 1)%i = [1:9 13:19 76:91 94:100]
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,~] = fileparts(file_path);
    mask = imread(file_path);
    mask = imbinarize(mask);
    %mask = imresize(mask, [512 512], "nearest");
    imwrite(mask, fullfile(output_path, [name '.tif']));
end
%%
classNames = {'Adipocyte'};
pixelLabelIDs = 1;

dsPred = pixelLabelDatastore(output_path,classNames,pixelLabelIDs);
dsGND = pixelLabelDatastore(gnd_datastore_path,classNames,pixelLabelIDs);
tic
metrics = evaluateSemanticSegmentation(dsPred,dsGND, "Verbose",true);
t = toc/60
%%
save_pth = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\evaluation';
save(fullfile(save_pth, 'evaluation_model_Ov1_MTC_aug_1024_intratumoral_fat_abdominal_laparoscopty_semseg.mat'),'metrics')
%% ROC Curve
precision = metrics.ClassMetrics.Precision{1,1}(1:end);
recall = metrics.ClassMetrics.Recall{1,1}(1:end);

[sorted_recall, sortOrder] = sort(recall);
    
    % Sort vectorB using the same sorting order as vectorA
    sorted_precision = precision(sortOrder);

plot(sorted_recall, sorted_precision)
title("ROC glands model x20")
ylabel("Recall")
xlabel("Precision")
%%
function out = predDataReader(data_path)

load(data_path)
if isempty(inst_id)
inst_map(inst_map == -1) = 0;
out{1} = logical(inst_map);
out{2} = categorical(repmat({'Adipocyte'}, 1, 1));
out{3} = 1;

else
inst_map(inst_map == 0) = inst_id(end)+1;
inst_map(inst_map == -1) = 0;
inst_id(inst_id == 0) = inst_id(end)+1;
[inst_id, exists_in_map ] = clean_ids(inst_map, inst_id);

mask = instancemask2maskstack(inst_map);
out{1} = mask;

N=size(inst_id, 1);
class_vector = categorical(repmat({'Adipocyte'}, N, 1));
out{2} = class_vector;


inst_scores(exists_in_map == 0) = [];
out{3} = inst_scores;
end
end

function out = gndDataReader(data_path)

class_map = imread(data_path);
inst_map = bwlabel(class_map, 4);
inst_maskstack = instancemask2maskstack(inst_map);
out{1} = inst_maskstack;

ids = unique(inst_map);
ids(ids == 0) = [];
N=size(ids, 1);
class_vector = categorical(repmat({'Adipocyte'}, N, 1));
out{2} = class_vector;

end

function [inst_id, exists_in_map ]= clean_ids(inst_map, inst_id)

map_ids = unique(inst_map);
map_ids(map_ids == 0) = [];

exists_in_map = ismember(inst_id, int32(map_ids));

% Remove elements from vector B that are not in vector A
inst_id(exists_in_map == 0) = [];

end
