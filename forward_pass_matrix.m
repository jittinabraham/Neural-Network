function output_forward_trained =forward_pass_matrix( network2, input_vec )
 numLayers = numel(network2);
   
    for i = 2:numLayers
        weights = network2(i).W;
        bias = network2(i).b;
        z = weights * input_vec + bias;
        input_vec = sigmoid(z);
       
        
    end
   
    output_forward_trained = input_vec;


end 

function output = sigmoid(x)
    output = 1 ./ (1 + exp(-x));
end