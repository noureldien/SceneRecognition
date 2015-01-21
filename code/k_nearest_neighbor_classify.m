% Starter code prepared by James Hays for CS 143, Brown University

%This function will predict the category for every test image by finding
%the training image with most similar features. Instead of 1 nearest
%neighbor, you can vote based on k nearest neighbors which will increase
%performance (although you need to pick a reasonable value for k).

function predicted_categories = k_nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, categories, neighbors)
% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

%{
Useful functions:
 matching_indices = strcmp(string, cell_array_of_strings)
   This can tell you which indices in train_labels match a particular
   category. Not necessary for simple one nearest neighbor classifier.

 D = vl_alldist2(X,Y)
    http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y.
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator '
   vl_alldist2 supports different distance metrics which can influence
   performance significantly. The default distance, L2, is fine for images.
   CHI2 tends to work well for histograms.
 
  [Y,I] = MIN(X) if you're only doing 1 nearest neighbor, or
  [Y,I] = SORT(X) if you're going to be reasoning about many nearest
  neighbors

%}

% loop on all the test features
% and for each one of them calculate the list of
% distances between it and the train features
% then sort all these distances ascendingly (shortest distance first)
% then according to how many k you have
% predict the label of this feature
% finally, return this list of predicted labels

train_N = size(train_image_feats, 1);
test_N = size(test_image_feats, 1);

predicted_categories = cell(test_N, 1);
predicted_categories_nums = zeros(test_N, 1);
K = neighbors;

for i=1:test_N
    
    distances = zeros(train_N,1);
    for j=1:train_N
        distances(j) = vl_alldist2(test_image_feats(i,:)', train_image_feats(j,:)', 'l2');
    end    
    
    % note that idx is the list of indeces
    % ironically, I only need the indeces of the sorted distances
    % for example: distances(idx(i)) = sorted_distances(i);
    [~, idx] = sort(distances);
    
    % now for the first k elements, get what are their labels
    % and see which is the most mentioned label in these k-elements
    k_labels = cell(K, 1);
    for j=1:K
        k_labels(j) = train_labels(idx(j));
    end
    k_labels_numbers = labels_numbers(k_labels, categories);
    
    % see which label was the most mentioned
    occurances = zeros(K, 1);
    for j = 1:K
        occurances(j) = sum(k_labels_numbers == k_labels_numbers(j));
    end
    [~, index] = ismember(max(occurances), occurances);
    predicted_categories{i} = k_labels{index};
    predicted_categories_nums(i) = k_labels_numbers(index);
    
    % for testing
    %disp(k_labels);
    %disp(k_labels_numbers);
    %disp(predicted_categories{i});
    %disp(predicted_categories_nums(i));
end

end















