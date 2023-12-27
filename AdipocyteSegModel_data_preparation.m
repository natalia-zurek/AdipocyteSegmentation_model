% ======== DATASET PREPARATION ========
img_pth = "E:\WSIs\Massons trichrome stain\Set-1\HE-20x\";
xml_pth = "E:\WSIs\Massons trichrome stain\Set-1\HE-20x\";
%%
output_path = 'C:\Projects\Adipocyte segmentation model\images';
mkdir(output_path);
img = dir(fullfile(img_pth, '*.svs'));

for i = 1:size(img, 1)
    img_pth_full = fullfile(img(i).folder, img(i).name);
    [~,name,~] = fileparts(img_pth_full);
    xml_pth_full = fullfile(xml_pth, [name '.xml']);

    he = CutRoIsFromFile(img_pth_full, xml_pth_full);
    save_images_from_cell(he, output_path, name)
end

%%
% RGB images that serve as input to the network, specified as H-by-W-by-3 numeric arrays.
% Bounding boxes for objects in the RGB images, specified as NumObjects-by-4 matrices, with rows in the format [x y w h]).
% 
% Instance labels, specified as NumObjects-by-1 string vectors.
% 
% Instance masks. Each mask is the segmentation of one instance in the image. 
% The COCO data set specifies object instances using polygon coordinates formatted as NumObjects-by-2 cell arrays. 
% Each row of the array contains the (x,y) coordinates of a polygon along the boundary of one instance in the image. 
% However, the Mask R-CNN in this example requires binary masks specified as logical arrays of size H-by-W-by-NumObjects.

%load masks
%bwconn
%get bboxes
%instance labels (1)

% images_path = 'C:\Projects\Adipocyte segmentation model\dataset\train';
% output_path = 'C:\Projects\Adipocyte segmentation model\dataset\final';
% mask_path = 'C:\Projects\Adipocyte segmentation model\dataset\adipocyte masks';

images_path = 'C:\Ovarian cancer project\Adipocyte dataset\images MTD';
out_folder = 'C:\Ovarian cancer project\Adipocyte dataset\train\deleted images';
output_path = 'C:\Ovarian cancer project\Adipocyte dataset\train\final_data_MTD';
mask_path = 'C:\Ovarian cancer project\Adipocyte dataset\train\masks';
mkdir(output_path);

files = [dir(fullfile(images_path, '*.tif')); dir(fullfile(images_path, '*.jpg')); dir(fullfile(images_path, '*.png'))];
%%
addpath(genpath('c:/Ovarian cancer project/AdipocyteSegmentation_model'));

save_overlay = 1;
save_dataset = 1;

for i = 1:size(files, 1)
    file_path = fullfile(files(i).folder, files(i).name);
    [~,name,ext] = fileparts(file_path);
    imageName = [name ext];
    
    mask_path_full = fullfile(mask_path, [name '.png']);
    if ~isfile(mask_path_full)
        disp(['Mask ' name '.png doesnt exist'])
        status = movefile(file_path, out_folder);
        continue
    end

    img = imread(file_path);
    mask = imread(mask_path_full);

    % prepare data
    bw_mask = bwlabel(mask);
    if save_overlay
    ov = labeloverlay(img, bw_mask);
    imwrite(ov, fullfile(output_path, [name '.png']))
    end

    if save_dataset
    props = regionprops("struct",bw_mask, 'BoundingBox');
    bbox = cat(1, props.BoundingBox);
    N=size(bbox,1);
    label = categorical(repmat({'Adipocyte'}, N,1));
    masks = instancemask2maskstack(bw_mask);

    save(fullfile(output_path, [name '.mat']),'imageName', "bbox", 'masks', 'label')
    end

end


%%
% imds = imageDatastore(images_path, "FileExtensions",[".jpg",".tif", ".png"]);
% imds.ReadFcn = @read_image;
% blds = boxLabelDatastore(T);
% mskds = imageDatastore(mask_path);
% mskds.ReadFcn = @read_instance_masks;
% ds = combine(imds, blds, mskds);



