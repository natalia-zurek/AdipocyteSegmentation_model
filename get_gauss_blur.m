function img_aug = get_gauss_blur(img)
min_sigma = 0.0001;
max_sigma = 1.0;  %1.3 to 2
sigma = (max_sigma-min_sigma)*rand(1,1)+min_sigma;   % par of gaussian filter
img_aug = imgaussfilt(img, sigma);

end