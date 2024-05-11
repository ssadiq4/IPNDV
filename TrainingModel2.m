imgWidth = 1940;
imgHeight = 1460;

%Loading Training Dataset
imds_Image = imageDatastore('DeepLearningData/TrainingData', "FileExtensions", ".mat", 'ReadFcn', @(x) double(load(x).ReturnArray{1}));
Boxes = datastore('DeepLearningData/TrainingData', 'Type', 'file', 'ReadFcn', @(x) (load(x).ReturnArray{2}));
Labels = datastore('DeepLearningData/TrainingData', 'Type', 'file', 'ReadFcn', @(x) categorical(load(x).ReturnArray{3}));
blds = boxLabelDatastore(table(readall(Boxes),readall(Labels)));
imds_Mask = imageDatastore ('DeepLearningData/TrainingData', "FileExtensions", ".mat", 'ReadFcn', @(x) load(x).ReturnArray{4});
TrainingData = combine(imds_Image, blds, imds_Mask);
preview(TrainingData)

test_Image = imageDatastore('DeepLearningData/TrainingData/TrainVal', "FileExtensions", ".mat", 'ReadFcn', @(x) double(load(x).ReturnArray{1}));
test_boxes = datastore('DeepLearningData/TrainingData/TrainVal', 'Type', 'file', 'ReadFcn', @(x) (load(x).ReturnArray{2}));
test_Labels = datastore('DeepLearningData/TrainingData/TrainVal', 'Type', 'file', 'ReadFcn', @(x) categorical(load(x).ReturnArray{3}));
test_blds = boxLabelDatastore(table(readall(test_boxes),readall(test_Labels)));
test_Mask = imageDatastore('DeepLearningData/TrainingData/TrainVal', "FileExtensions", ".mat", 'ReadFcn', @(x) load(x).ReturnArray{4});
valData = combine(test_Image,test_blds, test_Mask);

options = trainingOptions("sgdm", ...
    LearnRateSchedule="piecewise", ...
    InitialLearnRate = 0.01, ... 
    Plot="training-progress", ...
    MaxEpochs=10 , ...
    MiniBatchSize=3, ...
    ValidationData = valData, ...
    BatchNormalizationStatistics="moving", ...
    Shuffle="every-epoch",...
    ResetInputNormalization=false);

trainClassNames = ["Actin"];
detector = maskrcnn("resnet50-coco", trainClassNames);
doTraining = true;
% [net,info] = trainMaskRCNN(TrainingData,detector,options);
if doTraining
    [net,info] = trainMaskRCNN(TrainingData,detector,options);
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save("trainedMaskRCNN-"+modelDateTime+".mat","net");
end