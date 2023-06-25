function network = init_net(layerSizes )
   % layerSizes=[784 180 90 50 10];
   
    numLayers = numel(layerSizes) - 1;
    network = struct();
    
    for layer = 2:numLayers+1
        inputSize = layerSizes(layer-1);
        outputSize = layerSizes(layer);
        
        % Initialize weights and biases for the current layer
        network(layer).W = randn(outputSize, inputSize);
        network(layer).b = randn(outputSize, 1);
    end
end
