function save_images_from_cell(img_cell, output_path, name)

for i = 1:length(img_cell)
    imwrite(img_cell{i}, fullfile(output_path, [name '_' num2str(i) '.tif']))
end

end