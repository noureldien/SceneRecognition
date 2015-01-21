% Starter code prepared by James Hays and Sam Birch for CS 143, Brown U.

% All of your code will be in "Step 1" and "Step 2", although you can
% modify other parameters in the starter code.

%% Step 0: Set up parameters, vlfeat, category list, and image paths.

%For this project, you will need to report performance for three
%combinations of features / classifiers. It is suggested you code them in
%this order, as well:
% 1) Tiny image features and nearest neighbor classifier
% 2) Bag of sift features and nearest neighbor classifier
% 3) Bag of sift features and linear SVM classifier
%The starter code is initialized to 'placeholder' just so that the starter
%code does not crash when run unmodified and you can get a preview of how
%results are presented.

clc;

FEATURES = {'tiny image', 'bag of patch', 'bag of sift', 'bag of phow', 'placeholder'};
FEATURE = FEATURES{2};

CLASSIFIERS = {'k nearest neighbor', 'support vector machine', 'placeholder'};
CLASSIFIER = CLASSIFIERS{2};

%number of training examples per category to use. Max is 100. For
%simplicity, we assume this is the number of test cases per category, as
%well.
num_train_per_cat = 100;

% set up paths to VLFeat functions.
% See http://www.vlfeat.org/matlab/matlab.html for VLFeat Matlab documentation
% This should work on 32 and 64 bit versions of Windows, MacOS, and Linux
run('vlfeat/toolbox/vl_setup')

data_path = '../data/'; %change if you want to work with a network copy

%This is the list of categories / directories to use. The categories are
%somewhat sorted by similarity so that the confusion matrix looks more
%structured (indoor and then urban and then rural).
categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', ...
    'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street', ...
    'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'};

%This list of shortened category names is used later for visualization.
abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub', ...
    'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For'};

%This function returns cell arrays containing the file path for each train
%and test image, as well as cell arrays with the label of each train and
%test image. By default all four of these arrays will be 1500x1 where each
%entry is a char array (or string).
fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);
%   train_image_paths  1500x1   cell
%   test_image_paths   1500x1   cell
%   train_labels       1500x1   cell
%   test_labels        1500x1   cell

%% Step 1: Represent each image with the appropriate feature
% Each function to construct features should return an N x d matrix, where
% N is the number of paths passed to the function and d is the
% dimensionality of each image representation. See the starter code for
% each function for more details.

fprintf('Using %s representation for images\n', FEATURE);

switch lower(FEATURE)
    case 'tiny image'
        % best parameter values for svm classifier
        % to classify tiny-image
        svm_iterations = 100 * 1000;
        svm_lambda = 0.0001;
        % after many validations, this is the best k value
        % for k-NN classifier when tiny-image is used
        neighbors_count = 6;
        % YOU CODE get_tiny_images.m
        if ~exist('tiny_images.mat', 'file')
            fprintf('Bag of sifts does not exist for train/test data, Computing one from training/test images\n');
            train_image_feats = get_tiny_images(train_image_paths);
            test_image_feats = get_tiny_images(test_image_paths);
            save('tiny_images.mat', 'train_image_feats', 'test_image_feats');
        else
            load('tiny_images.mat', 'train_image_feats', 'test_image_feats');
        end
    case 'bag of patch'
        % best parameter values for svm classifier
        % to classify bag-of-patch
        svm_iterations = 100 * 1000;
        svm_lambda = 0.00001;
        % after many validations, this is the best k value
        % for k-NN classifier when bag-of-patch us used
        neighbors_count = 15;
        % YOU CODE build_patch_vocabulary.m
        if ~exist('patch_vocab.mat', 'file')
            fprintf('No existing visual word vocabulary found. Computing one from training images\n');
            %Larger values will work better (to a point) but be slower to compute
            patch_vocab_size = 100;
            patch_size = 8;
            patch_stride = 24;
            patch_vocab = build_patch_vocabulary(train_image_paths, patch_vocab_size, patch_size, patch_stride);
            save('patch_vocab.mat', 'patch_vocab');
        else
            load('patch_vocab.mat', 'patch_vocab');
        end
        % YOU CODE get_bags_of_patch.m
        patch_size = 8;
        patch_stride = 16;
        if ~exist('image_patch_feats.mat', 'file')
            fprintf('Bag of patches does not exist for train/test data, Computing one from training/test images\n');
            train_image_feats = get_bags_of_patch(train_image_paths, patch_vocab, patch_size, patch_stride);
            test_image_feats = get_bags_of_patch(test_image_paths, patch_vocab, patch_size, patch_stride);
            save('image_patch_feats.mat', 'train_image_feats', 'test_image_feats');
        else
            load('image_patch_feats.mat', 'train_image_feats', 'test_image_feats');
        end
    case 'bag of sift'
        % best parameter values for svm classifier
        % to classify bag-of-sift
        svm_iterations = 100 * 1000;
        svm_lambda = 0.00001;
        % after many validations, this is the best k value
        % for k-NN classifier when bag-of-sift us used
        neighbors_count = 15;
        % YOU CODE build_sift_vocabulary.m
        if ~exist('sift_vocab.mat', 'file')
            fprintf('No existing visual word vocabulary found. Computing one from training images\n');
            %Larger values will work better (to a point) but be slower to compute
            sift_vocab_size = 100;
            sift_steps = 100;
            sift_vocab = build_sift_vocabulary(train_image_paths, sift_vocab_size, sift_steps);
            save('sift_vocab.mat', 'sift_vocab');
        else
            load('sift_vocab.mat', 'sift_vocab');
        end
        % YOU CODE get_bags_of_sifts.m
        sift_steps = 20;
        if ~exist('image_sift_feats.mat', 'file')
            fprintf('Bag of sifts does not exist for train/test data, Computing one from training/test images\n');
            train_image_feats = get_bags_of_sifts(train_image_paths, sift_steps, sift_vocab);
            test_image_feats = get_bags_of_sifts(test_image_paths, sift_steps, sift_vocab);
            save('image_sift_feats.mat', 'train_image_feats', 'test_image_feats');
        else
            load('image_sift_feats.mat', 'train_image_feats', 'test_image_feats');
        end
    case 'bag of phow'
        % best parameter values for svm classifier
        % to classify bag-of-phow
        svm_iterations = 10 * 1000;
        svm_lambda = 0.0002;
        % after many validations, this is the best k value
        % for k-NN classifier when bag-of-phow us used
        neighbors_count = 15;
        % YOU CODE build_phow_vocabulary.m
        if ~exist('phow_vocab.mat', 'file')
            fprintf('No existing visual word vocabulary found. Computing one from training images\n');
            %Larger values will work better (to a point) but be slower to compute
            phow_vocab_size = 100;
            phow_steps = 100;
            phow_vocab = build_phow_vocabulary(train_image_paths, phow_vocab_size, phow_steps);
            save('phow_vocab.mat', 'phow_vocab');
        else
            load('phow_vocab.mat', 'phow_vocab');
        end
        % YOU CODE get_bags_of_phow.m
        phow_steps = 20;
        if ~exist('image_phow_feats.mat', 'file')
            fprintf('Bag of PHOW does not exist for train/test data, Computing one from training/test images\n');
            train_image_feats = get_bags_of_phow(train_image_paths, phow_steps, phow_vocab);
            test_image_feats = get_bags_of_phow(test_image_paths, phow_steps, phow_vocab);
            save('image_phow_feats.mat', 'train_image_feats', 'test_image_feats');
        else
            load('image_phow_feats.mat', 'train_image_feats', 'test_image_feats');
        end
    case 'placeholder'
        train_image_feats = [];
        test_image_feats = [];
    otherwise
        error('Unknown feature type');
end

fprintf('Finish calculating (%s\n) features for training/test data\n', FEATURE);

% If you want to avoid recomputing the features while debugging the
% classifiers, you can either 'save' and 'load' the features as is done
% with vocab.mat, or you can utilize Matlab's "code sections" functionality
% http://www.mathworks.com/help/matlab/matlab_prog/run-sections-of-programs.html

%% Step 2: Classify each test image by training and using the appropriate classifier
% Each function to classify test features will return an N x 1 cell array,
% where N is the number of test cases and each entry is a string indicating
% the predicted category for each test image. Each entry in
% 'predicted_categories' must be one of the 15 strings in 'categories',
% 'train_labels', and 'test_labels'. See the starter code for each function
% for more details.

fprintf('Using %s classifier to predict test set categories\n', CLASSIFIER);

switch lower(CLASSIFIER)
    case 'k nearest neighbor'
        % YOU CODE k_nearest_neighbor_classify.m
        predicted_categories = k_nearest_neighbor_classify( ...
            train_image_feats, train_labels, test_image_feats, ...
            categories, neighbors_count);
        
    case 'support vector machine'
        % YOU CODE svm_classify.m
        predicted_categories = svm_classify(train_image_feats, ...
            train_labels, test_image_feats, categories, ...
            svm_lambda, svm_iterations);
        
    case 'placeholder'
        %The placeholder classifier simply predicts a random category for
        %every test case
        random_permutation = randperm(length(test_labels));
        predicted_categories = test_labels(random_permutation);
        
    otherwise
        error('Unknown classifier type');
end

fprintf('Finish calculating (%s\n) classifier for training/test data\n', CLASSIFIER);

%% Step 3: Build a confusion matrix and score the recognition system
% You do not need to code anything in this section.

% If we wanted to evaluate our recognition method properly we would train
% and test on many random splits of the data. You are not required to do so
% for this project.

% This function will recreate results_webpage/index.html and various image
% thumbnails each time it is called. View the webpage to help interpret
% your classifier performance. Where is it making mistakes? Are the
% confusions reasonable?
create_results_webpage( ...
    train_image_paths, ...
    test_image_paths, ...
    train_labels, ...
    test_labels, ...
    categories, ...
    abbr_categories, ...
    predicted_categories)

% Interpreting your performance with 100 training examples per category:
%  accuracy  =   0 -> Your code is broken (probably not the classifier's
%                     fault! A classifier would have to be amazing to
%                     perform this badly).
%  accuracy ~= .07 -> Your performance is chance. Something is broken or
%                     you ran the starter code unchanged.
%  accuracy ~= .20 -> Rough performance with tiny images and nearest
%                     neighbor classifier. Performance goes up a few
%                     percentage points with K-NN instead of 1-NN.
%  accuracy ~= .20 -> Rough performance with tiny images and linear SVM
%                     classifier. The linear classifiers will have a lot of
%                     trouble trying to separate the classes and may be
%                     unstable (e.g. everything classified to one category)
%  accuracy ~= .50 -> Rough performance with bag of SIFT and nearest
%                     neighbor classifier. Can reach .60 with K-NN and
%                     different distance metrics.
%  accuracy ~= .60 -> You've gotten things roughly correct with bag of
%                     SIFT and a linear SVM classifier.
%  accuracy >= .70 -> You've also tuned your parameters well. E.g. number
%                     of clusters, SVM regularization, number of patches
%                     sampled when building vocabulary, size and step for
%                     dense SIFT features.
%  accuracy >= .80 -> You've added in spatial information somehow or you've
%                     added additional, complementary image features. This
%                     represents state of the art in Lazebnik et al 2006.
%  accuracy >= .85 -> You've done extremely well. This is the state of the
%                     art in the 2010 SUN database paper from fusing many
%                     features. Don't trust this number unless you actually
%                     measure many random splits.
%  accuracy >= .90 -> You get to teach the class next year.
%  accuracy >= .96 -> You can beat a human at this task. This isn't a
%                     realistic number. Some accuracy calculation is broken
%                     or your classifier is cheating and seeing the test
%                     labels.




