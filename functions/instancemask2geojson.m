function FC = instancemask2geojson(instance_mask)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

idx = unique(instance_mask);
idx(idx == 0) = [];
P = cell(length(idx),1);

for i = 1:length(idx)
    bw_mask = zeros(size(instance_mask));
    bw_mask(instance_mask == idx(i)) = 1;
    B = bwboundaries(bw_mask, 4);  
    P{i,1} = [B{1}(:,2), B{1}(:,1)];

end
keepIndex = cellfun(@(x) size(x, 1) > 3, P);
    P_filtered = P(keepIndex);

    FC= contour2geojson(P_filtered, 'Tumor');

% %B = bwboundaries(instance_mask, 4, 'noholes');
% B = boundarymask(instance_mask, 4);  
% P = B;
%     for k = 1:length(B)
%         boundary = B{k};
%         P{k} = [boundary(:,2), boundary(:,1)];
%     end
%     keepIndex = cellfun(@(x) size(x, 1) > 3, P);
%     P_filtered = P(keepIndex);
% 
%     FC= contour2geojson(P_filtered, 'Adipocyte');
end