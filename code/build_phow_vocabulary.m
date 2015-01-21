% Starter code prepared by James Hays for CS 143, Brown University

%This function will sample PHOW descriptors from the training images,
%cluster them with kmeans, and then return the cluster centers.

function vocab = build_phow_vocabulary( image_paths, vocab_size, phow_steps )

% The inputs are images, a N x 1 cell array of image paths and the size of
% the vocabulary.

% The output 'vocab' should be vocab_size x 128. Each row is a cluster
% centroid / visual word.

% Load images from the training set. To save computation time, you don't
% necessarily need to sample from all images, although it would be better
% to do so. You can randomly sample the descriptors from each image to save
% memory and speed up the clustering. Or you can simply call vl_phow with
% a large step size here, but a smaller step size in make_hist.m.

% For each loaded image, get some PHOW features. You don't have to get as
% many PHOW features as you will in get_bags_of_phow.m, because you're only
% trying to get a representative sample here.

% Once you have tens of thousands of PHOW features from many training
% images, cluster them with kmeans. The resulting centroids are now your
% visual word vocabulary.

% file name where phow features extracted
% from train set should be saved
phow_feats_file_name = 'build_vocabulary_phow_feats.mat';

% check if exist, load, if not, run
if exist(phow_feats_file_name, 'file')
    phow_feats = zeros(128, 0);
    load(phow_feats_file_name, 'phow_feats');
else
    % steps at which sift of phow is extracted
    steps = phow_steps;
    % list of the phow features of all images
    phow_feats = zeros(128, 0);
        
    % loop on all the images
    N = length(image_paths);
    
    % Get the phow features
    for i=1:N
        % get cropped image
        img_cropped = get_image_cropped(image_paths{i});
        % convet to single precision for the sake of vl_phow
        img_single = single(img_cropped);
        % get the phow features and accumulate them to the list
        [~, img_phow_feats] = vl_phow(img_single, 'Step', steps);
        phow_feats = [phow_feats img_phow_feats];
    end
    % save phow features
    save(phow_feats_file_name, 'phow_feats');
end

% after calculating phow, cluster them into k clusters
% using k-means
phow_feats = single(phow_feats);

% notice that the centroid matrix is the needed vocabulary
% the K (number of clusters) is given to the function
% and it is the size of vocabulary (vocab_size)
% C [128 * K]
% phow [128 * num-of-features-for-all-images ]
K = vocab_size;
[C, ~] = vl_kmeans(phow_feats, K, 'distance', 'l1', 'algorithm', 'elkan');
vocab = C;

% or we can use matlab k-means
% sure to transpose the cetriod matrix because vocab
%[~, C] = kmeans(x_train, K);
%C = C';

end





