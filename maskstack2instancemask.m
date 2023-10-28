function instance_mask = maskstack2instancemask(maskstack)

num_stack = size(maskstack, 3);
instance_mask = zeros(size(maskstack, 1), size(maskstack, 2));

for i = 1:num_stack

instance_mask(maskstack(:,:,i) == 1) = i;

end

end