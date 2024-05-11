% Assuming 'metrics' is already loaded into the workspace
% If not, load from a file:
if ~exist('metrics', 'var')
    metrics = load('evaluationMetrics.mat').metrics;
end

% Displaying Metrics in the Command Window
disp('Dataset Metrics:');
disp(metrics.DatasetMetrics);

disp('Class Metrics:');
disp(metrics.ClassMetrics);

disp('Image Metrics:');
disp(metrics.ImageMetrics);

% Confusion Matrix Visualization
figure;
confusionchart(metrics.ConfusionMatrix);
disp(metrics.ConfusionMatrix);

title('Confusion Matrix');
xlabel('Predicted Class');
ylabel('True Class');

% Normalized Confusion Matrix Visualization
figure;
confusionchart(metrics.NormalizedConfusionMatrix);
title('Normalized Confusion Matrix');
xlabel('Predicted Class');
ylabel('True Class');

% Class-specific Metrics Visualization
figure;
subplot(2,1,1);
bar(metrics.ClassMetrics.Precision{:});
disp(metrics.ClassMetrics.Precision{:});
title('Precision by Class');
xticklabels(metrics.ClassNames);
ylabel('Precision');

subplot(2,1,2);
bar(metrics.ClassMetrics.Recall{:});
title('Recall by Class');
xticklabels(metrics.ClassNames);
ylabel('Recall');

% Plotting IoU Histogram
ious = extractIoUsFromMetrics(metrics); % Assuming you have a function to extract IoUs
figure;
histogram(ious, 'BinWidth', 0.1);
title('Histogram of Intersection over Union (IoU)');
xlabel('IoU');
ylabel('Frequency');

% Display mean Average Precision if available
if isfield(metrics.DatasetMetrics, 'mAP')
    disp(['Mean Average Precision (mAP): ', num2str(metrics.DatasetMetrics.mAP)]);
end

% Save plots if necessary
savePlot('Confusion Matrix', 'confusion_matrix.png');
savePlot('Normalized Confusion Matrix', 'normalized_confusion_matrix.png');
savePlot('Precision by Class', 'precision_by_class.png');
savePlot('Recall by Class', 'recall_by_class.png');
savePlot('IoU Histogram', 'iou_histogram.png');

% Helper function to save figures
function savePlot(figTitle, fileName)
    fig = findobj('Type', 'figure', 'Name', figTitle);
    if ~isempty(fig)
        saveas(fig, fullfile('figures', fileName));
    end
end

% Assuming a custom function to extract IoUs from your metrics
function ious = extractIoUsFromMetrics(metrics)
    % Dummy data: replace with actual extraction logic
    ious = rand(100, 1); % Random data for demonstration
end
