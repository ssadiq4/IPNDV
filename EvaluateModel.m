test_images = imageDatastore('DeepLearningData/ValidationData', "FileExtensions", ".mat", 'ReadFcn', @(x) double(load(x).ReturnArray{1}));

test_Array = zeros(1940,1460,3,7);
for i = 1:7
    test_Array(:,:,:,i) = readimage(test_images,i);
end 

pretrained = load("trainedMaskRCNN-2024-05-08-23-32-39.mat");
net = pretrained.net;

[masks, labels, scores] = segmentObjects(net,test_Array);

mask_Data = datastore(masks);
label_Data = datastore(labels);
scores_Data = datastore(scores);
predictedData = combine(mask_Data,label_Data,scores_Data);

truth_Mask = imageDatastore ('DeepLearningData/ValidationData', "FileExtensions", ".mat", 'ReadFcn', @(x) load(x).ReturnArray{4});
truth_Label = datastore('DeepLearningData/ValidationData', 'Type', 'file', 'ReadFcn', @(x) categorical(load(x).ReturnArray{3}));
truthData = combine(truth_Mask,truth_Label);


metrics = evaluateInstanceSegmentation(predictedData,truthData);
preview(metrics)