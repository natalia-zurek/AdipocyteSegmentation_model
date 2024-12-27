I = imread("C:\Users\wylezoln\Box\_my_projects\Kidney project\NataliaZ\1_C.png");
load("C:\Users\wylezoln\Box\_my_projects\Kidney project\NataliaZ\ATmask\1_C_Mask.mat");
%%
ATmask(ATmask == 17) = 0;
ATmask(ATmask == 7) = 0;
%%
ov = labeloverlay(I, ATmask, "Colormap", [0 0.5 1; 0 0 1], "Transparency", 0.5);
imshow(ov)
%%
imwrite(ov, '001_pat_mask.png')
%%
imwrite(mask, "001_maskov.png")