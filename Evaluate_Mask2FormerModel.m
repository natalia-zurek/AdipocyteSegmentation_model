ground_truth_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\predictions\mask2former_adipocyte_test_epoch_80 laparoscopy\mat';
prediction_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\predictions\mask2former_adipocyte_test_epoch_80 laparoscopy\mat';
output_path = 'C:\Ovarian cancer project\Adipocyte dataset\Mask2Former\evaluation\test';
mkdir(output_path);

files = dir(fullfile(ground_truth_path, '*.mat'));
for i = 1%:size(files, 1)
    ground_truth_fullpath = fullfile(files(i).folder, files(i).name);
    prediction_fullpath = fullfile(prediction_path, files(i).name);
    
    gnd = load(ground_truth_fullpath);
    pred = load(prediction_fullpath);

    metrics = evaluateInstanceSegmentation(pred.inst_map,gnd.inst_map);

end
