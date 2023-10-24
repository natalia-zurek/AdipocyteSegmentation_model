function instance_masks = read_instance_masks(filename)

mask = imread(filename);
bw_mask = bwlabel(mask);
instance_masks = get_instance_masks(bw_mask);

end