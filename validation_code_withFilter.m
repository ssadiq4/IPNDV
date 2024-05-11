% Load test images from .mat files
valImds = imageDatastore('DeepLearningData/ValidationData_mod', ...
    "FileExtensions", ".mat", 'ReadFcn', @(x) double(load(x).ReturnArray{1}));

% Load the pretrained Mask R-CNN model
pretrained = load("trainedMaskRCNN-2024-05-08-23-32-39.mat");
net = pretrained.net;

% Prepare an array to hold all images if multiple images need processing
numImages = numel(valImds.Files);
test_Array = zeros(1940, 1460, 3, numImages, 'like', readimage(valImds, 1));
validIndices = [];  % Store indices of successfully processed images

% Load images into an array
for i = 1:numImages
    test_Array(:,:,:,i) = readimage(valImds, i);
end 

results = cell(numImages, 3);  % Pre-allocate cell array for results

% Perform segmentation using the network directly on the image array
for i = 1:numImages
    try
        [masks, labels, scores] = segmentObjects(net, test_Array(:,:,:,i), 'Threshold', 0.2);
        results{i, 1} = masks; 
        results{i, 2} = labels;
        results{i, 3} = scores;
        validIndices = [validIndices, i];  % Append index if processing was successful
    catch exception
        disp(['Error processing image ' num2str(i) ': ' exception.message]);
        % Do not add index to validIndices
    end
end

% Filter results and ground truth data to include only successfully processed images
filteredResults = results(validIndices, :);
dsResults = arrayDatastore(filteredResults, 'OutputType', 'same');
transformedDS = transform(dsResults, @(x) {x{1}, x{2}, x{3}});

% Filter corresponding ground truth data
truth_Masks = imageDatastore('DeepLearningData/ValidationData_mod', ...
    "FileExtensions", ".mat", 'ReadFcn', @(x) logical(load(x).ReturnArray{4}));
truth_Labels = imageDatastore('DeepLearningData/ValidationData_mod', ...
    "FileExtensions", ".mat", 'ReadFcn', @(x) categorical(load(x).ReturnArray{3}));

% Manually set the files for truth datastores to the files corresponding to valid indices
truth_Masks.Files = truth_Masks.Files(validIndices);
truth_Labels.Files = truth_Labels.Files(validIndices);
truthData = combine(truth_Masks, truth_Labels);

% Evaluate the instance segmentation performance
metrics = evaluateInstanceSegmentation(transformedDS, truthData, 0.5, 'Verbose', true);

% Display the evaluation metrics
disp(metrics);

% Save the evaluation metrics to a file
save('evaluationMetrics.mat', 'metrics');
