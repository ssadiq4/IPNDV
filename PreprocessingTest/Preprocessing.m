I = imread('Cell_01_Actin.tif');
I = imclearborder(I); %ADDED THIS
[~,threshold] = edge(I,'sobel');
fudgeFactor = 0.4; %changed from 0.5 to 0.4 
BWs = edge(I,'sobel',threshold * fudgeFactor);
se90 = strel('line',5,90); %changed from 3 to 5
se0 = strel('line',5,0);
BWsdil = imdilate(BWs,[se90 se0]);
BWdfill = imfill(BWsdil,'holes');
BWnobord = imclearborder(BWdfill,4);
seD = strel('diamond',1);
BWfinal = imerode(BWnobord,seD);
BWfinal = imerode(BWfinal,seD);
%% new stuff
BWfinal = bwareafilt(BWfinal,1);
I(~BWfinal) = 0;
I = adapthisteq(I);
imshow(I)