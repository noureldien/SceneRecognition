% Starter code prepared by James Hays for CS 143, Brown University

%This function will sample SIFT descriptors from the training images,
%cluster them with kmeans, and then return the cluster centers.

function vocab = build_sift_vocabulary( image_paths, vocab_size, sift_steps )

% The inputs are images, a N x 1 cell array of image paths and the size of
% the vocabulary.

% The output 'vocab' should be vocab_size x 128. Each row is a cluster
% centroid / visual word.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img)
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be thrown away here
  (but possibly used for extra credit in get_bags_of_sifts if you're making
  a "spatial pyramid").
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

[centers, assignments] = vl_kmeans(X, K)
 http://www.vlfeat.org/matlab/vl_kmeans.html
  X is a d x M matrix of sampled SIFT features, where M is the number of
   features sampled. M should be pretty large! Make sure matrix is of type
   single to be safe. E.g. single(matrix).
  K is the number of clusters desired (vocab_size)
  centers is a d x K matrix of cluster centroids. This is your vocabulary.
   You can disregard 'assignments'.

  Matlab has a build in kmeans function, see 'help kmeans', but it is
  slower.
%}

% Load images from the training set. To save computation time, you don't
% necessarily need to sample from all images, although it would be better
% to do so. You can randomly sample the descriptors from each image to save
% memory and speed up the clustering. Or you can simply call vl_dsift with
% a large step size here, but a smaller step size in make_hist.m.

% For each loaded image, get some SIFT features. You don't have to get as
% many SIFT features as you will in get_bags_of_sift.m, because you're only
% trying to get a representative sample here.

% Once you have tens of thousands of SIFT features from many training
% images, cluster them with kmeans. The resulting centroids are now your
% visual word vocabulary.

% file name where sift features extracted
% from train set should be saved
sift_feats_file_name = 'build_vocabulary_sift_feats.mat';

% check if exist, load, if not, run
if exist(sift_feats_file_name, 'file')
    sift_feats = zeros(128, 0);
    load(sift_feats_file_name, 'sift_feats');
else
    % the more the steps, the more sift features are extracted
    % steps = 10 gives ~ 800K features for all images
    %                  ~ 48.5K featurs for 1 image
    steps = sift_steps;
    % list of the sift features of all images
    sift_feats = zeros(128, 0);
    
    % loop on all the images
    N = length(image_paths);
    
    % Get the sift features
    for i=1:N
        % get cropped image
        img_cropped = get_image_cropped(image_paths{i});
        % convet to single precision for the sake of vl_dsift
        img_single = single(img_cropped);
        % get the sift features and accumulate them to the list
        % notice that dense sift is used
        [~, img_sift_feats] = vl_dsift(img_single, 'fast', 'Size', 3, 'Step', steps);
        sift_feats = [sift_feats img_sift_feats];
    end
    % save sift features
    save(sift_feats_file_name, 'sift_feats');
end

% after calculating sift, cluster them into k clusters
% using k-means
sift_feats = single(sift_feats);

% notice that the centroid matrix is the needed vocabulary
% the K (number of clusters) is given to the function
% and it is the size of vocabulary (vocab_size)
% C [128 * K]
% sift_feats [128 * num-of-features-for-all-images ]
K = vocab_size;
[C, ~] = vl_kmeans(sift_feats, K, 'distance', 'l1', 'algorithm', 'elkan');
vocab = C;

% or we can use matlab k-means
% sure to transpose the cetriod matrix because vocab
%[~, C] = kmeans(x_train, K);
%C = C';

end





