% Starter code prepared by James Hays for CS 143, Brown University

%This feature representation is described in the handout, lecture
%materials, and Szeliski chapter 14.

function image_feats = get_bags_of_sifts(image_paths, sift_steps, vocab)

% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an 128 x N
% matrix 'vocab' where each column is a kmeans centroid or visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every time at significant expense.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram.

% You will want to construct SIFT features here in the same way you
% did in build_vocabulary.m (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% SIFT features will look very different from a smaller version of the same
% image.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img)
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be used for extra
  credit if you are constructing a "spatial pyramid".
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

D = vl_alldist2(X,Y)
   http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y.
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator '  You can use this to figure out the closest
    cluster center for every SIFT feature. You could easily code this
    yourself, but vl_alldist2 tends to be much faster.

Or:

For speed, you might want to play with a KD-tree algorithm (we found it
reduced computation time modestly.) vl_feat includes functions for building
and using KD-trees.
 http://www.vlfeat.org/matlab/vl_kdtreebuild.html
%}

% vocab size is 128*D
% 128 is the length of sift feature
% D is the K/clusters of k-means/size of visual word/size of vocabulary
[~, D] = size(vocab);

% extract sift features for all the given images
N = size(image_paths, 1);
steps = sift_steps;

% the required return
image_feats = zeros(N,D);

for i=1:N
    % get cropped image
    img_cropped = get_image_cropped(image_paths{i});
    % convet to single precision for the sake of vl_dsift
    img_single = single(img_cropped);
    % get the sift features for the image
    % notice that dense sift is used
    [~, img_sift_feats] = vl_dsift(img_single, 'fast', 'Size', 3, 'Step', steps);
    % single the sift feats for the sake of distance measurement
    img_sift_feats = single(img_sift_feats);
    % for each local feature of the sift features of one image
    % assign it to its nearest cluster, then put all
    % these assignments in another matrix call it histogram
    % which tells you how many times a centroid is used for assignment
    % then normalize this histogram
    % well, this histogram then is the needed feature to this image
    img_histo = zeros(1, D);
    Q = size(img_sift_feats, 2);
    distances = zeros(D, 1);
    for j=1:Q
        % for each local feature, get the distances to the centroids
        for k=1:D
            distances(k) = vl_alldist2(img_sift_feats(:,j), vocab(:,k), 'l1');
        end
        % then get the index of the nearest centroid (i.e. nearest distance)
        [~, index] = ismember(min(distances), distances);
        % and increment it to the histogram
        img_histo(index) = img_histo(index) + 1;        
    end
    % normalize the histogram, this is
    % the wrong way: img_histo = img_histo ./ sum(img_histo)
    % while the next is the correct one
    img_histo = img_histo/norm(img_histo);    
    % add it to the features matrix
    image_feats(i,:) = img_histo;
end

end




