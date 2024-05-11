% Load test images from .mat files
valImds = imageDatastore('DeepLearningData/ValidationData_mod', ...
    "FileExtensions", ".mat", 'ReadFcn', @(x) double(load(x).ReturnArray{1}));

% Load the pretrained Mask R-CNN model
pretrained = load("trainedMaskRCNN-2024-05-08-23-32-39.mat");
net = pretrained.net;

% Prepare an array to hold all images if multiple images need processing
numImages = numel(valImds.Files);
test_Array = zeros(1940, 1460, 3, numImages, 'like', readimage(valImds, 1));

% Load images into an array
for i = 1:numImages
    test_Array(:,:,:,i) = readimage(valImds, i);
end 

% Perform segmentation using the network directly on the image array
[masks, labels, scores] = segmentObjects(net, test_Array, 'Threshold', 0.2);
%disp(masks{i});
disp(labels);
% Prepare the results in the required format
results = cell(numImages, 3);
for i = 1:numImages
    disp("here");

    results{i, 1} = (masks{i}); % Individual mask
    results{i, 2} = labels{i}; % Corresponding label
    results{i, 3} = scores{i}; % Corresponding score
end

% Convert results to a format suitable for evaluation
dsResults = arrayDatastore(results, 'OutputType', 'same');

% Transform dsResults to fit the format expected by evaluateInstanceSegmentation
transformedDS = transform(dsResults, @(x) {x{1}, x{2}, x{3}});

% Load ground truth data
truth_Masks = imageDatastore('DeepLearningData/ValidationData_mod', ...
    "FileExtensions", ".mat", 'ReadFcn', @(x) logical(load(x).ReturnArray{4}));
truth_Labels = imageDatastore('DeepLearningData/ValidationData_mod', ...
    "FileExtensions", ".mat", 'ReadFcn', @(x) categorical(load(x).ReturnArray{3}, standardCategories));
truthData = combine(truth_Masks, truth_Labels);

% Evaluate the instance segmentation performance
metrics = evaluateInstanceSegmentation(transformedDS, truthData, 0.5, 'Verbose', true);

% Display the evaluation metrics
disp(metrics);

% Save the evaluation metrics to a file
save('evaluationMetrics.mat', 'metrics');
