function resized_image = read_image(filename)

img = imread(filename);
resized_image = imresize(img, [800, 800], "bilinear");
end