function FC = adiposoft2geojson(instance_mask)
%ADIPOSOFT2GEOJSON function that converts instance mask from Adiposoft to
%FeatureCollection that can be save as geojson file compatible with QuPath

idx = unique(instance_mask);
idx(idx == 0) = [];
P = cell(length(idx),1);

k=1;
for i = 1:length(idx)
    bw_mask = zeros(size(instance_mask));
    bw_mask(instance_mask == idx(i)) = 1;
    B = bwboundaries(bw_mask, 4);
    for j = 1:size(B, 1)
        P{k,1} = [B{j}(:,2), B{j}(:,1)];
        k = k+1;
    end


end
keepIndex = cellfun(@(x) size(x, 1) > 3, P);
P_filtered = P(keepIndex);

FC = contour2geojson(P_filtered, 'Adipocyte');

end