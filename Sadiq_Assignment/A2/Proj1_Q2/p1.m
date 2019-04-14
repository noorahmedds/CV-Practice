src = imread('Audrey.jpg');
frame = imread('Frame.jpg');

% SRC of the math:
% https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
% https://medium.com/the-andela-way/foundations-of-machine-learning-singular-value-decomposition-svd-162ac796c27d

% First we need to determine the four correspondances in src and frame
src_x = [0, size(src,1), 0, size(src,1)];
src_y = [0, 0, size(src,2), size(src,2)];
    
imshow(frame);
[frame_x, frame_y] = getpts;

% Generate the P vector
P = generateP(src_x, src_y, frame_x, frame_y);
[U,S,V] = svd(P);

% Create target vector M2
v = ones(8, 1);
for i = (1:4)
    v((2*(i-1)) + 1) = frame_x(i);
    v((2*(i)) + 1) = frame_y(i);
end

H = inv(P' * P) * ( P' * v);

% V ka singular matrix chahiyay
% H = calcLeftSingularVector(V);

applyHomography(H, src, frame);

% Estimate Homography H using SVD. Get the last singular vector of V == H
% Update values from the H*src(x,y) onto the frame

function applyHomography(H, src, frame)
    for i = size(src, 1)
        for j = size(src,2)
            currPixel = H * [i, j, 1];
        end
    end
end

function output = generateP(src_x, src_y, frame_x, frame_y)
%    create a 2x8 matrix
    
    output = zeros(2*4, 8);
%     {xA, yA, 1, 0, 0, 0, -xA*xB, -yA*xB}
%     {0, 0, 0, xA, yA, 1, -xA*yB, -yA*yB}

    for i = (1:4)
        a = [src_x(i), src_y(i), 1, 0, 0, 0, -src_x(i) * frame_x(i), -src_y(i) * frame_y(i)]
        b = [0,0,0, src_x(i), src_y(i), 1, -src_x(i)*frame_y(i),src_y(i) * frame_y(i)]
        output((2*(i-1)) + 1, :) = a;
        output((2*(i)) + 1, :) = b;
    end
end

function applyHomography(H, src, frame)

end