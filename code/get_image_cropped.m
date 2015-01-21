
% get image from the given path then crop it from the center
% so it becomes a square image
function img_cropped = get_image_cropped(path)

% read image
img = imread(path);
% crop it according to shortest dimention/side
[img_h, img_w] = size(img);
min_side = min([img_h img_w]);
% crop it
crop_rect = [floor((img_w - min_side)/2), floor((img_h - min_side)/2),  min_side, min_side];
img_cropped = imcrop(img, crop_rect);

end