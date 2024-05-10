% Load the trained model
loadedData = load("trainedMaskRCNN-2024-05-09-21-13-04.mat");
detector = loadedData.net;
standardCategories = ["Actin"];  % Define your standard categories for validation

% Setup the directory and create datastores for validation images
valImageDir = 'DeepLearningData/ValidationData';
valImds = imageDatastore(valImageDir, "FileExtensions", ".mat", 'ReadFcn', @(x) double(load(x).ReturnArray{1}));

% Assuming predictions need to be saved to temporary files
tempDir = 'tempResults';
if ~exist(tempDir, 'dir')
    mkdir(tempDir);
end

% Process each image and save predictions as files
for i = 1:9
    img = readimage(valImds, i);
    fprintf('Processing image %d of %d\n', i, numel(valImds.Files));
    [predictedMasks, scores, labels] = segmentObjects(detector, img, 'MiniBatchSize', 1);

    % Cast types before saving
    predictedMasks = logical(predictedMasks);  % Ensure masks are logical
    if isnumeric(labels)  % Convert numeric labels to categorical
        labels = categorical(labels, 1:numel(standardCategories), standardCategories);
    else
        labels = categorical(labels, standardCategories);  % Ensure labels are categorical
    end
    scores = double(scores);  % Ensure scores are numeric

    save(fullfile(tempDir, sprintf('result_%d.mat', i)), 'predictedMasks', 'scores', 'labels');
end

disp('Segmentation and detection completed.');

% Function to read results from files
function data = readResult(file)
    standardCategories = ["Actin"];
    loaded = load(file);
    predictedMasks = logical(loaded.predictedMasks);
    labels = categorical(loaded.labels, standardCategories);  % Ensure labels are categorical and conform to defined categories
    scores = double(loaded.scores);
    data = {predictedMasks, labels, scores};
end

% Create custom datastores for predicted results
predictedDS = fileDatastore(fullfile(tempDir, 'result_*.mat'), 'ReadFcn', @readResult, 'FileExtensions', '.mat');

% Load Ground Truth Data
truthMasks = imageDatastore(valImageDir, "FileExtensions", ".mat", 'ReadFcn', @(x) logical(load(x).ReturnArray{4}));
truthLabels = imageDatastore(valImageDir, "FileExtensions", ".mat", 'ReadFcn', @(x) categorical(load(x).ReturnArray{3}, standardCategories));
truthData = combine(truthMasks, truthLabels);

% Specify overlap threshold and options
threshold = 0.5;
options.Verbose = true;
% Code to ensure that both predicted and truth data have matching entries
numPredictedFiles = numel(predictedDS.Files);
numTruthFiles = numel(truthMasks.Files);  % Assuming truthMasks and truthLabels have the same number of files
disp(['Number of predicted files: ', num2str(numel(predictedDS.Files))]);
disp(['Number of ground truth files: ', num2str(numel(truthMasks.Files))]);

if numPredictedFiles ~= numTruthFiles
    error('Mismatch in the number of predicted and ground truth files');
end
% Evaluate instance segmentation
metrics = evaluateInstanceSegmentation(predictedDS, truthData, threshold, Verbose=true);

% Display the metrics
disp(metrics);

% Save metrics to a MAT file
save('evaluation_metrics.mat', 'metrics');

% Cleanup temporary files
rmdir(tempDir, 's');
