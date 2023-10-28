function FC = instancemask2geojson(instance_mask)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

B = bwboundaries(instance_mask,4, 'noholes');
    P = B;
    for k = 1:length(B)
        boundary = B{k};
        P{k} = [boundary(:,2), boundary(:,1)];
    end
    keepIndex = cellfun(@(x) size(x, 1) > 3, P);
    P_filtered = P(keepIndex);

    FC= contour2geojson(P_filtered, 'Adipocyte');
end