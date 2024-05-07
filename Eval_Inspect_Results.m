%Define and hold test images 
dataSetDir = fullfile(toolboxdir("vision"),"visiondata","triangleImages");
testImagesDir = fullfile(dataSetDir,"testImages");
imds = imageDatastore(testImagesDir);

%Define and label ground truths
testLabelsDir = fullfile(dataSetDir,"testLabels");
classNames = ["triangle" "background"];
labelIDs = [255 0];
pxdsTruth = pixelLabelDatastore(testLabelsDir,classNames,labelIDs);

%Run the semantic segmentation classifier
net = load("triangleSegmentationNetwork.mat");
net = net.net;
pxdsResults = semanticseg(imds,net,Classes=classNames,WriteLocation=tempdir);

%Evaluate the quality of the prediction
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);

%Inspect class metrics
metrics.ClassMetrics

%Display confusion matrix
metrics.ConfusionMatrix
cm = confusionchart(metrics.ConfusionMatrix.Variables, ...
  classNames,Normalization="row-normalized");
cm.Title = "Normalized Confusion Matrix (%)";