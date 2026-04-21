clc;
trained_images=('/MATLAB Drive/data/');
filenames=dir(fullfile(trained_images,'*.j'));

imds = imageDatastore(trained_images,'IncludeSubfolders',true,'LabelSource','foldernames');
labelCount = countEachLabel(imds);
img = readimage(imds,1);
size(img)


figure; 
perm = randperm(50,20); 
for i = 1:20   
    
    subplot(5,4,i);   
    imshow(imds.Files{perm(i)});
    %imresize(imds,[120 120])
end

numTrainFiles = 80; 
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize'); 
layers = [     
    imageInputLayer([64 64 1])          
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer     
    reluLayer         
    maxPooling2dLayer(2,'Stride',2)       
    convolution2dLayer(3,32,'Padding','same')    
    batchNormalizationLayer  
    reluLayer       
    maxPooling2dLayer(2,'Stride',2)    
    convolution2dLayer(3,64,'Padding','same') 
    batchNormalizationLayer  
    reluLayer 
    maxPooling2dLayer(2,'Stride',2)    
    convolution2dLayer(3,128,'Padding','same') 
    batchNormalizationLayer  
    reluLayer 
    fullyConnectedLayer(6) 
    softmaxLayer  
    classificationLayer
    ]; 
options = trainingOptions('sgdm','InitialLearnRate',0.01,'MaxEpochs',180,'Shuffle','every-epoch','ValidationData',imdsValidation,'ValidationFrequency',26,'Verbose',false,'Plots','training-progress');
net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels; 

accuracy = sum(YPred == YValidation)/numel(YValidation);

 plotconfusion(YValidation,YPred);