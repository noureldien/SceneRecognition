% Starter code prepared by James Hays for CS 143, Brown University

%This feature is inspired by the simple tiny images used as features in
%  80 million tiny images: a large dataset for non-parametric object and
%  scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
%  Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
%  pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

function image_feats = get_tiny_images(image_paths, file_name)
% image_paths is an N x 1 cell array of strings where each string is an
%  image path on the file system.
% image_feats is an N x d matrix of resized and then vectorized tiny
%  images. E.g. if the images are resized to 16x16, d would equal 256.

% To build a tiny image feature, simply resize the original image to a very
% small square resolution, e.g. 16x16. You can either resize the images to
% square while ignoring their aspect ratio or you can crop the center
% square portion out of each image. Making the tiny images zero mean and
% unit length (normalizing them) will increase performance modestly.

% suggested functions: imread, imresize

D = 16;
N = length(image_paths);
image_feats = zeros(N,(D^2));

for i=1:N
    % Create Tiny Image
    % get path
    path = image_paths{i};
    % read image
    img = imread(path);
    % crop it according to shortest dimention/side
    [img_h, img_w] = size(img);
    min_side = min([img_h img_w]);
    % crop it
    crop_rect = [floor((img_w - min_side)/2), floor((img_h - min_side)/2),  min_side, min_side];
    img_cropped = imcrop(img, crop_rect);
    % resize it
    img_resized = imresize(img_cropped, [D D]);
    % convet to 1D array
    % note that reshape() concatenate matrix columns
    % and we want to concatenate matrix rows, that's why
    % img_resized is transposed before reshaped
    img_array = reshape(img_resized',1,[]);
    image_feats(i,:) = img_array;
    
    % for testing
    %     figure(1); clf;
    %     imshow(img);
    %     daspect([1 1 1]);
    %
    %     figure(2); clf;
    %     imshow(img_cropped);
    %     daspect([1 1 1]);
    %
    %     daspect([1 1 1]);
    %     figure(3); clf;
    %     imshow(img_resized);
    %     daspect([1 1 1]);
    %
    %     disp('the resized image');
    %     disp(img_resized);
    %     disp('the array');
    %     disp(img_array);    
    
    
end





