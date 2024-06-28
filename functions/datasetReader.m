function  out = datasetReader(filename, images_path)

load(filename,"imageName","bbox","label","masks");

im = imread(fullfile(images_path, imageName));

out{1} = im;
out{2} = bbox;
out{3} = label;
out{4} = masks;

end