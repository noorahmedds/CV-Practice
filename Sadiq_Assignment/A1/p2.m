% The intuition is that we want to retain a pixel where if a pixel to its
% 125 degree angle is on then we want to succesively subtract its value
% from it. Its a 5x5 matrix because we want to have atleast 3 pixel wide
% diagonal

kernel = [-2, -1,   0,    0,   0; 
          -1,  0,   0,    0,   0;
           0,  0,   1,    0,   0;
           0,  0,   0,    0,  -1;
           0,  0,   0,   -1,  -2;]
      
im = imread('P1_1/Lines.png');

im;

newIm = imfilter(im, kernel);
imshow(newIm);