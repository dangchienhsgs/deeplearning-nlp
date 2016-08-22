nn = require("nn")
nninit = require("nninit")
require 'optim'
csvigo =  require("csvigo")
require 'torch'



-- load trainning_size
matrix_words = csvigo.load{path='pre-processing/train_data/full_train_matrix_file.txt', mode='large', separator=' '}
classes = {"O", "B-LOC", "B-PER", "B-ORG", "B-TOUR", "I-ORG", "I-PER", "I-TOUR", "I-LOC", "B-PRO", "I-PRO"}

windowSize = #matrix_words[1]
trainingSize = #matrix_words

data = torch.Tensor(trainingSize, windowSize)
for i=1, trainingSize do
    for j=1, windowSize do
        data[i][j] = tonumber(matrix_words[i][j]) + 1 -- lua index start from 1, not 0
    end
end


-- read labels
labelsRaw = csvigo.load{path='pre-processing/train_data/full_train_label_file.txt', mode='large', separator=' '}
labels = torch.DoubleTensor(#labelsRaw)
for i=1, #labelsRaw do    
    labels[i] = tonumber(labelsRaw[i][1]) + 1
end

print(string.format("window_size = %d, training_size = %d", windowSize, trainingSize))



-- load test_data -- 
matrix_words_test = csvigo.load{path='pre-processing/train_data/full_test_matrix_file.txt', mode='large', separator=' '}
testingSize = #matrix_words_test

data_test = torch.Tensor(testingSize, windowSize)
for i=1, testingSize do
    for j=1, windowSize do        
        data_test[i][j] = tonumber(matrix_words_test[i][j]) + 1 -- lua index start from 1, not 0
    end
end


-- read labels -- 
labelsRawTest = csvigo.load{path='pre-processing/train_data/full_test_label_file.txt', mode='large', separator=' '}
labels_test = torch.DoubleTensor(#labelsRawTest)
for i=1, #labelsRawTest do    
     labels_test[i] = tonumber(labelsRawTest[i][1]) + 1
end

print(string.format("window_size = %d, testing_size = %d", windowSize, testingSize))

-- load embedded
w2v_mat = csvigo.load{path='pre-processing/train_data/total_matrix_final.txt', mode='large', separator=' '}

rows = #w2v_mat
cols = #w2v_mat[1] - 1 -- the last elem is \n

-- load word
words = csvigo.load{path='pre-processing/train_data/total_words_final.txt', mode='large', separator=' '}
dictSize = #words
embeddedSize = cols

print (string.format("Size {rows=%d, cols=%d}", rows, cols))
print (string.format("Number of total words = %d", dictSize))



-- create tensor --
csv_tensor = torch.Tensor(rows, cols)
print(string.format("Generate new tensor at size %d x %d", rows, cols))

for i=1, rows do
    for j=1, cols do                 
        csv_tensor[i][j] = tonumber(w2v_mat[i][j])
    end
end


lookup = nn.LookupTable(dictSize, embeddedSize)
lookup.weight = csv_tensor

print(#lookup:forward(data[1]))



-- some matrix layer --
L = 100
K = embeddedSize

windowSize = 11
model = nn.Sequential()
model:add(lookup)
model:add(nn.Reshape(K*windowSize))
model:add(nn.Linear(K*windowSize, L))
model:add(nn.HardTanh())
model:add(nn.Linear(L, #classes))
model:add(nn.LogSoftMax())

print(model)


-- generate weight for criterion --
weight = torch.Tensor(#classes)
for i=1, #classes do 
    weight[i] = 0
end

for i=1, labels:size()[1] do    
    weight[labels[i]] = weight[labels[i]] + 1
end

for i=1, #classes do
    weight[i] = labels:size()[1] / weight[i]
end

print(weight)



criterion = nn.ClassNLLCriterion(weight)
x, dl_dx = model:getParameters()

sgd_params = {
   learningRate = 1e-2,
   learningRateDecay = 1e-4,
   weightDecay = 1e-3,
   momentum = 1e-4
}


print(trainingSize)
step = function(batch_size)
    local current_loss = 0
    local count = 0
    local shuffle = torch.randperm(trainingSize)    
    batch_size = batch_size or 200
    
    for t = 0, trainingSize, batch_size do
        if t>=trainingSize then
            break
        end
        
        -- setup inputs and targets for this mini-batch
        local size = math.min(t + batch_size, trainingSize) - t        
        local inputs = torch.Tensor(size, windowSize)
        local targets = torch.Tensor(size)
        for i = 1, size do            
            local input = data[shuffle[i+t]]
            local target = labels[shuffle[i+t]]            
            -- if target == 0 then target = 10 end
            inputs[i] = input
            targets[i] = target
        end
        
        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end
            dl_dx:zero()

            -- perform mini-batch gradient descent            
            outputs = model:forward(inputs);            
            local loss = criterion:forward(model:forward(inputs), targets)        
            model:backward(inputs, criterion:backward(model.output, targets))

            return loss, dl_dx
        end
        
        _, fs = optim.sgd(feval, x, sgd_params)
        -- fs is a table containing value of the loss function
        -- (just 1 value for the SGD optimization)
        count = count + 1
        current_loss = current_loss + fs[1]        
   end

    -- normalize loss
    return current_loss / count
end

print(testingSize)
eval = function(batch_size)
    local count = 0
    batch_size = batch_size or 200        
    
    true_prob = {}
    false_prob = {}
    
    -- init true and false count --
    for i=1, #classes do
        true_prob[i] = 0
        false_prob[i] = 0
    end
    
    for i = 0, testingSize, batch_size do                
        if i >= testingSize then
            break
        end
        
        local size = math.min(i + batch_size, testingSize) - i        
        local inputs = data_test[{{i+1,i+size}}]
        local targets = labels_test[{{i+1,i+size}}]:long()
        local outputs = model:forward(inputs)    
        local _, indices = torch.max(outputs, 2)                
        
        guessed_right = 0                
        for j=1, indices:size()[1] do
            label = targets[j]    
            if indices[j][1] == targets[j] then
                guessed_right = guessed_right + 1                             
                true_prob[label] = true_prob[label] + 1
            else 
                false_prob[label] = false_prob[label] + 1
            end
        end                
        
        count = count + guessed_right                    
    end
    
    return count/testingSize, true_prob, false_prob
end

max_iters = 5
do
    local last_accuracy = 0
    local decreasing = 0
    local threshold = 1 -- how many deacreasing epochs we allow
    for iter = 1, max_iters do
        local loss = step(20)        
        print(string.format('Epoch: %d Current loss: %4f', iter, loss))
        local accuracy, true_prob, false_prob = eval(20)        
        print(string.format('Accuracy on the validation set: %4f', accuracy))
        for x, y in pairs(true_prob) do
            print(string.format('Count label %d number of true=%d, number of false=%d, accuracy=%4f', x, y, false_prob[x], y/(y+false_prob[x])))
        end
        
        if accuracy < last_accuracy then
             if decreasing > threshold then break end
             decreasing = decreasing + 1
        else
             decreasing = 0
        end
        last_accuracy = accuracy
    end
end