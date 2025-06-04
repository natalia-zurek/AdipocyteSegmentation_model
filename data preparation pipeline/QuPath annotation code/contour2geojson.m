function [FC] = contour2geojson(polygons, class_name)
%createGeojson transform polygons and levels from contour to geojson file
%   Detailed explanation goes here
% % TO SAVE
% fileID = fopen(fullfile(save_path, [name, '.geojson']),'w');
% fwrite(fileID,jsonencode(FC));
% fclose(fileID)
% generate geojson
FC={};
FC.type="FeatureCollection";
FC.features={};

    for i = 1:size(polygons,1)
        F={};
        F.type='Feature';
        F.geometry={};
        F.properties.objectType="annotation";
        F.properties.classification.name = class_name;
        F.properties.classification.color = [1 0 0];
        F.geometry.type="Polygon";
        coor = polygons{i};
        coor = [coor; coor(1,1:end)];
        F.geometry.coordinates = {coor};
        FC.features=[FC.features;F];
        coor = [];
    end
end