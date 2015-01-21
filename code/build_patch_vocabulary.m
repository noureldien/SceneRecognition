
%This function will sample patch descriptors from the training images,
%cluster them with kmeans, and then return the cluster centers.

function vocab = build_patch_vocabulary( image_paths, vocab_size, patch_size, patch_stride )

% The inputs are images, a N x 1 cell array of image paths and the size of
% the vocabulary.

% The output 'vocab' should be vocab_size x patch_size. Each row is a cluster
% centroid / visual word.

% Load images from the training set. To save computation time, you don't
% necessarily need to sample from all images, although it would be better
% to do so. You can randomly sample the descriptors from each image to save
% memory and speed up the clustering. Or you can simply call vl_phow with
% a large step size here, but a smaller step size in make_hist.m.

% For each loaded image, get some patch features. You don't have to get as
% many patch features as you will in get_bags_of_patch.m, because you're only
% trying to get a representative sample here.

% Once you have tens of thousands of patch features from many training
% images, cluster them with kmeans. The resulting centroids are now your
% visual word vocabulary.

% file name where patch features extracted
% from train set should be saved
patch_feats_file_name = 'build_vocabulary_patch_feats.mat';

% patch size
M = patch_size;

% check if exist, load, if not, run
if exist(patch_feats_file_name, 'file')
    patch_feats = zeros(M, 0);
    load(patch_feats_file_name, 'patch_feats');
else
    % list of the patch features of all images
    patch_feats = zeros(M * M, 1000*1500*2);
    row = 1;
    
    % how many pixel we move after taking the patch
    stride = patch_stride;
    
    % loop on all the images
    N = length(image_paths);
    
    % Get the patch features
    for i=1:N
        disp(i);
        % get image
        img = imread(image_paths{i});
        [img_h, img_w] = size(img);
        y = 1;
        x = 1;
        % get the patch features and accumulate them to the list
        while (y <= img_h - stride)
            while (x <= img_w - stride)
                % extract patch
                img_patch_feats = img(y:y+M-1, x:x+M-1);
                % convet to 1D array
                % note that reshape() concatenate matrix columns
                % and we want to concatenate matrix rows, that's why
                % img_resized is transposed before reshaped
                img_patch_feats = reshape(img_patch_feats',1,[]);
                patch_feats(:, row) = img_patch_feats';
                row = row + 1;
                x = x + stride;
            end
            x = 1;
            y = y + stride;
        end
    end
    
    % resize and save patch features
    patch_feats = patch_feats(:, 1:row-1);
    save(patch_feats_file_name, 'patch_feats');
end

% after calculating patch, cluster them into k clusters
% using k-means
patch_feats = single(patch_feats);

% notice that the centroid matrix is the needed vocabulary
% the K (number of clusters) is given to the function
% and it is the size of vocabulary (vocab_size)
% C [M * K]
% patch [M * num-of-features-for-all-images ]
K = vocab_size;
[C, ~] = vl_kmeans(patch_feats, K, 'distance', 'l1', 'algorithm', 'elkan');
vocab = C;

% or we can use matlab k-means
% sure to transpose the cetriod matrix because vocab
%[~, C] = kmeans(x_train, K);
%C = C';

end





