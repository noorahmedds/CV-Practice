im = imread('cameraman.tif');
[Gx, Gy] = imgradientxy(im);
[Gmag, Gdir] = imgradient(Gx, Gy);

% Show the gradients as a 2D map
imshow(Gdir);