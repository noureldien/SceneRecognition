% Starter code prepared by James Hays for CS 143, Brown University

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters.

function predicted_categories = svm_classify(train_image_feats, ...
    train_labels, test_image_feats, categories, lambda, iterations)
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
  category. This is useful for creating the binary labels for each SVM
  training task.

[W B] = vl_svmtrain(features, labels, LAMBDA)
  http://www.vlfeat.org/matlab/vl_svmtrain.html

  This function trains linear svms based on training examples, binary
  labels (-1 or 1), and LAMBDA which regularizes the linear classifier
  by encouraging W to be of small magnitude. LAMBDA is a very important
  parameter! You might need to experiment with a wide range of values for
  LAMBDA, e.g. 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10.

  Matlab has a built in SVM, see 'help svmtrain', which is more general,
  but it obfuscates the learned SVM parameters in the case of the linear
  model. This makes it hard to compute "confidences" which are needed for
  one-vs-all classification.

%}

cat_N = length(categories);
train_N = size(train_labels, 1);
test_N = size(test_image_feats, 1);

categories_nums = linspace(1, 15, 15);
predicted_categories = cell(test_N, 1);
train_labels_nums = labels_numbers(train_labels, categories);
scores = zeros(test_N, cat_N);

% loop on all categories
for i=1:cat_N
    
    % get positive and negative images, positives
    % are the ones assigned to the current category
    % actually, we'll get only the index to these features
    idx_p = find(train_labels_nums == categories_nums(i));
    idx_n = find(train_labels_nums ~=(categories_nums(i)));
    
    % now construct the birary y which is an array
    % contains either 1 or -1 according to the previous indexes
    y = zeros(train_N, 1);
    for j = 1:length(idx_p)
        y(idx_p(j)) = 1;
    end
    for j = 1:length(idx_n)
        y(idx_n(j)) = -1;
    end
    
    % learn the linear classifier
    % solver: sdca, sgd
    % 'epsilon', 1e-3
    [w, b, ~] = vl_svmtrain(train_image_feats', y', lambda, ...
        'MaxNumIterations', iterations, ...
        'loss', 'LOGISTIC', ...
        'epsilon', 0.0001 );
    
    % now, if we applied the decision hyper-plane we have [w, b]
    % we can classify whether a test feature is either the current
    % category or not (we're only interested if it is the current category)
    % add the predictions for the current category
    scores(:,i) = (test_image_feats*w) + b;    
end

% loop on all the scores
for i=1:test_N
    
    % assign the test_image to the category which
    % has the biggest score
    score = scores(i,:);
    [~, idx] = ismember(max(score),score);    
    predicted_categories{i} = categories{idx};
    
end

end




