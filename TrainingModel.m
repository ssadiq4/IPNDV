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

options = trainingOptions("sgdm", ...
    InitialLearnRate=0.001, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=1, ...m
    LearnRateDropFactor=0.95, ...
    Plot="training-progress", ...
    Momentum=0.9, ...
    MaxEpochs=2 , ...
    MiniBatchSize=4, ...
    BatchNormalizationStatistics="moving", ...3
    ResetInputNormalization=false, ...
    VerboseFrequency=50);

trainClassNames = ["Actin"];
detector = maskrcnn("resnet50-coco", trainClassNames);
doTraining = true;
% [net,info] = trainMaskRCNN(TrainingData,detector,options);
if doTraining
    [net,info] = trainMaskRCNN(TrainingData,detector,options);
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save("trainedMaskRCNN-"+modelDateTime+".mat","net");
end