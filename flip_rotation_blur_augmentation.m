function [img_aug, mask_aug] = flip_rotation_blur_augmentation(img, mask, option)
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here

switch option
    case 'r90'
        img_aug = imrotate(img, 90, "bilinear");
        mask_aug = imrotate(mask, 90, "nearest");
    case 'r180'
        img_aug = imrotate(img, 180, "bilinear");
        mask_aug = imrotate(mask, 180, "nearest");
    case 'r270'
        img_aug = imrotate(img, 270, "bilinear");
        mask_aug = imrotate(mask, 270, "nearest");
    case 'vflip'
        img_aug = fliplr(img);
        mask_aug = fliplr(mask);
    case 'hflip'
        img_aug = flipud(img);
        mask_aug = flipud(mask);
    % case 'vhflip'
    %     img_aug = rot90(img,2);
    %     mask_aug = rot90(mask,2);
    case 'mblur'
        img_aug = medfilt3(img);
        mask_aug = mask;
    case 'gblur'
        img_aug = get_gauss_blur(img);
        mask_aug = mask;
    otherwise 
        disp('No such option')

end


end