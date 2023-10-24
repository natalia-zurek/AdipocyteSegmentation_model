function instance_masks = get_instance_masks(mask)

inst_id = unique(mask);
inst_id(inst_id == 0) = [];

num_inst = size(inst_id, 1);
%instance_masks = cell(num_inst,1);

instance_masks = false(size(mask));
instance_masks(mask == inst_id(1)) = true;
%instance_masks = imresize(instance_masks, [800, 800], 'nearest');

for i = 2:num_inst

    instance_mask = false(size(mask));
    instance_mask(mask == inst_id(i)) = true;
    %instance_mask = imresize(instance_mask, [800, 800], 'nearest');

    instance_masks = cat(3, instance_masks, instance_mask);
end

end