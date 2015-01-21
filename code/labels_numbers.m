
% convert list of labels to list of numbers according to the index
% of the label in the categories list
function labels_nums = labels_numbers(labels, categories)

labels_n = size(labels, 1);
labels_nums = zeros(labels_n, 1);
for i = 1:labels_n
    [~, index] = ismember(labels{i}, categories);
    labels_nums(i) = index;
end

end