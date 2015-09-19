
require("vocabularyBuilder")
require("utilities")
require "torch"
require "nn"
require "sys"
require 'optim'   
local LSTM = require 'LSTM'             -- LSTM timestep and utilities
local LSTM_Decoder = require 'LSTM_Decoder'             -- LSTM timestep and utilities
require 'beamSearch'             -- LSTM timestep and utilities
local model_utils=require 'model_utils'
local w_init=require("weight-init")

-- Use Float instead of Double
torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a Neural Network for End-to-end Sequence Translation')
cmd:text()
cmd:text('Options')
cmd:option('-train',"","Path to Train File -- Source and Target Sequence on same line separated by Tab")
cmd:option('-tune',"","Path to Development File -- Source and Target Sequence on same line separated by Tab")
cmd:option('-test',"","Path to Test File -- Source and Target Sequence on same line separated by Tab")
cmd:option('-source',"","Path to Full Source File for building Vocabulary")
cmd:option('-target',"","Path to Full Target File for building Vocabulary")
cmd:option('-hiddenSize',200,"Size of Hidden Layer Neurons")
cmd:option('-embeddingSize',50,"Size of Word Embeddings")
cmd:option('-predict',false,"Perform Only Prediction, no training")
cmd:text()

local options = cmd:parse(arg)

print("using Full Source File "..options.source)
print("using Full Target File "..options.target)
print("using Train File "..options.train)
print("using Tune File "..options.tune)
print("using Test File "..options.test)
print("using Hidden Layer Neurons "..options.hiddenSize)
print("using Embedding Size "..options.embeddingSize)

--Create Source Vocabulary and collect other information--
local sourceDictionary, reverseSourceDictionary, sourceMaxLength, sourceDictionarySize = createVocabulary(options.source)
print("Max Source Sequence Length "..sourceMaxLength.." and vocabulary size "..sourceDictionarySize)

--Create Target Vocabulary and collect other information--
local targetDictionary, reverseTargetDictionary, targetMaxLength, targetDictionarySize, EOS = createVocabulary(options.target)
print("Max Target Sequence Length "..targetMaxLength.." and vocabulary size "..targetDictionarySize)

local protos = {}
local function w_init_xavier_caffe(fan_in, fan_out)
   return math.sqrt(1/fan_in)
end

-- Send the input through a Lookup Table first to obtain it's embeddings
protos.embed = nn.Sequential():add(nn.LookupTable(sourceDictionarySize, options.embeddingSize)):add(nn.Reshape(options.embeddingSize))
-- LSTM Encoder takes the current input and embeddings and encodes the sequence seen till now
protos.lstm = LSTM.lstm(options.embeddingSize, options.hiddenSize)
-- LSTM Decoder takes the summary from LSTM Encoder generates the target sequence
protos.lstm_out = LSTM_Decoder.lstm(targetDictionarySize+options.hiddenSize,options.hiddenSize)
-- Use LogSoftmax to obtain probabilities for training
protos.softmax = nn.Sequential():add(nn.Linear(options.hiddenSize, targetDictionarySize)):add(nn.LogSoftMax())
protos.exp = nn.Sequential():add(nn.Exp())
protos.criterion = nn.ClassNLLCriterion()

-- put the above things into one flattened parameters tensor
local params, grad_params = model_utils.combine_all_parameters( protos.embed, protos.lstm,protos.lstm_out, protos.softmax)
params:uniform(-0.08, 0.08)

local method = "xavier_caffe"
protos.embed  = require('weight-init')(protos.embed, method)
protos.softmax = require('weight-init')(protos.softmax, method)
local params, grad_params = model_utils.combine_all_parameters( protos.embed, protos.lstm,protos.lstm_out, protos.softmax)

-- Use SGD with Nesterov Momentum
optimState = {
      learningRate = 0.01, -- Best was 0.1
      weightDecay = 0.0001,
      momentum = 0.001, -- Best was 0.1
      learningRateDecay = 0.0,
      nesterov = true,
      dampening = 0.0
   }
optimMethod = optim.sgd

-- make a bunch of clones, AFTER flattening, as that reallocates memory
local clones = {}
for name,proto in pairs(protos) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times(proto, math.max(targetMaxLength,sourceMaxLength)+1, not proto.parameters)
end

-- LSTM initial state (zero initially)
local initstate_c = torch.zeros( options.hiddenSize)
local initstate_h = initstate_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
local dfinalstate_c = initstate_c:clone()
local dfinalstate_h = initstate_c:clone()

-- LSTM initial state (zero initially)
local initstate_c_out = torch.zeros( options.hiddenSize)
local initstate_h_out = initstate_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
local dfinalstate_c_out = initstate_c:clone()
local dfinalstate_h_out = initstate_c:clone()


function test(testFileName,epoch)
  local f1 = lines_from(testFileName)
  io.close()
  
  local f = assert(io.open("output/test_out_"..epoch, "w"))

  local cost = 0.0
  local lineCount = 0
  
  for k,line in pairs(f1) do
      local splitWord = split(line,"\t")
      for index,word in ipairs(splitWord) do
        if index == 1 then
          sourceWord = word
        else
          targetWord = word
        end
      end
      
      f:write(sourceWord.."\t"..targetWord.."\t")
      print(sourceWord.."\t"..targetWord.."\t")
      
      local sourceLength = 0
      for word in string.gmatch(sourceWord,"[^ ]+") do
        sourceLength = sourceLength +1 
      end
      sourceLength = sourceLength +1
      
      local targetLength = 0
      for word in string.gmatch(targetWord,"[^ ]+") do
        targetLength = targetLength +1 
      end
      targetLength = targetLength +1
      
      local input= torch.Tensor(sourceLength,1):zero()
      local target = torch.Tensor(targetLength):zero()
      
      --read source and target sequence 
      local currentLength = 0;
      local index = 1;
      for word in string.gmatch(sourceWord,"[^ ]+") do
        if(sourceDictionary[word] == nil) then
          print(word.." Not found in source vocabulary")
        else
          index = sourceDictionary[word]
          input[currentLength + 1][1] = index
        end
        currentLength = currentLength +1; 
      end
      
      index = sourceDictionary["</S>"]
      input[currentLength + 1][1] = index 
      
      local embeddings = {}            -- input embeddings
      local lstm_c = {[0]=initstate_c} -- internal cell states of Encoder LSTM
      local lstm_h = {[0]=initstate_h} -- output values of Encoder LSTM
      
      local lstm_c_out = {[0]=initstate_c_out} -- internal cell states of Decoder LSTM1
      local lstm_h_out = {[0]=initstate_h_out} -- output values of Decoder LSTM1
      
      local predictions = {}           -- softmax outputs
      local loss = 0

      -- Send the input sequence through an LSTM Encoder
      for t=1,input:size(1) do
         embeddings[t] = clones.embed[t]:forward(input[t])[1]
         lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})
      end
      
      -- Last Hidden State of Encoder is the summary of the Source Sequence 
      local summary = lstm_h[input:size(1)]:clone()
      summary = summary:reshape(summary:size(1))
      
      local predictedSequence,cost = doBeamSearch(summary, clones, options, targetDictionarySize, EOS, reverseTargetDictionary)
      
      local predictedOutput = {}
      for i=1, #predictedSequence do
        local outputWord = reverseTargetDictionary[predictedSequence[i]]
        f:write(outputWord.." ")
        table.insert(predictedOutput, outputWord)
      end
      
      print(predictedOutput)
      print(cost)
      
      f:write("\n")
    end
    f:close()
end


function train(trainFileName, validFileName)
  local previousCost = 0.0
  local epoch = 0
  
--  For every sweep of training data
  while epoch < 200 do
    print("==> doing epoch "..epoch.." on training data with eta :"..optimState.learningRate)
    nClock = os.clock() 
    
    for line in io.lines(trainFileName) do
    
      -- extract every word --
      local sourceWord;
      local targetWord;
      
      -- Remember source and target sequence are separated by tab
      local splitWord = split(line,"\t")
      for index,word in ipairs(splitWord) do
        if index == 1 then
          sourceWord = word
        else
          targetWord = word
        end
      end
      
      local sourceLength = 0
      for word in string.gmatch(sourceWord,"[^ ]+") do
        sourceLength = sourceLength +1 
      end
      sourceLength = sourceLength +1
      
      local targetLength = 0
      for word in string.gmatch(targetWord,"[^ ]+") do
        targetLength = targetLength +1 
      end
      targetLength = targetLength +1
      
      local input= torch.Tensor(sourceLength,1):zero()
      local target = torch.Tensor(targetLength):zero()
      
      --read source and target sequence 
      local currentLength = 0;
      local index = 1;
      for word in string.gmatch(sourceWord,"[^ ]+") do
        if(sourceDictionary[word] == nil) then
          print(word.." Not found in source vocabulary")
        else
          index = sourceDictionary[word]
          input[currentLength + 1][1] = index
        end
        currentLength = currentLength +1; 
      end
      
      index = sourceDictionary["</S>"]
      input[currentLength + 1][1] = index * 1.0
      
      currentLength = 0;
      index = 1;
      for word in string.gmatch(targetWord,"[^ ]+") do
        if(targetDictionary[word] == nil) then
          print(word.." Not found in target vocabulary")
        else
          index = targetDictionary[word]
          target[currentLength + 1] = index 
        end
        currentLength = currentLength +1; 
      end
      
      index = targetDictionary["</S>"]
      target[currentLength + 1] = index
      
      local feval = function(params_)
          if params_ ~= params then
              params:copy(params_)
          end
          grad_params:zero()
    
           ------------------- forward pass -------------------
          local embeddings = {}            -- input embeddings
          local lstm_c = {[0]=initstate_c} -- internal cell states of Encoder LSTM
          local lstm_h = {[0]=initstate_h} -- output values of Encoder LSTM
          
          local lstm_c_out = {[0]=initstate_c_out} -- internal cell states of Decoder LSTM1
          local lstm_h_out = {[0]=initstate_h_out} -- output values of Decoder LSTM1
          
          local predictions = {}           -- softmax outputs
          local inputLSTM_Out = {}
          local loss = 0

          -- Send the input sequence through an LSTM Encoder
          for t=1,input:size(1) do
             embeddings[t] = clones.embed[t]:forward(input[t])[1]
             lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward({embeddings[t], lstm_c[t-1], lstm_h[t-1]}))
          end
          
          -- Last Hidden State of Encoder is the summary of the Source Sequence
          local summary = lstm_h[input:size(1)]:clone()
          summary = summary:reshape(summary:size(1))
          
          -- Use a Decoder for Prediction target sequence one by one
          for t=1,target:size(1) do
            -- Previous prediction is zero for time t = 1
            if t == 1 then
              inputLSTM_Out[t] = torch.cat(torch.zeros(targetDictionarySize), summary)
              lstm_c_out[t], lstm_h_out[t] = unpack(clones.lstm_out[t]:forward({inputLSTM_Out[t], lstm_c_out[t-1], lstm_h_out[t-1]}))
            else 
              local temp = clones.exp[t]:forward(predictions[t-1]):clone()
              inputLSTM_Out[t] = torch.cat(temp, summary)
              lstm_c_out[t], lstm_h_out[t] = unpack(clones.lstm_out[t]:forward({inputLSTM_Out[t], lstm_c_out[t-1], lstm_h_out[t-1]}))
            end
          
            predictions[t] = clones.softmax[t]:forward(lstm_h_out[t])
            
            loss = loss + clones.criterion[t]:forward(predictions[t]:float(), target[t])
          end

          ------------------ backward pass -------------------
          -- complete reverse order of the above
          local dembeddings = {}                              -- d loss / d input embeddings
          local dlstm_c = {[input:size(1)]=dfinalstate_c}    -- internal cell states of LSTM
          local dlstm_h = {}                                  -- output values of LSTM
          
          local dSummary = {}                              -- d loss / d summary
          local dTempSummary = {}                              -- d loss / d summary
          local dlstm_c_out = {[target:size(1)]=dfinalstate_c_out}    -- internal cell states of LSTM
          local dlstm_h_out = {}                                  -- output values of LSTM
          
          for t=target:size(1),1,-1 do
            local doutput_t = clones.criterion[t]:backward(predictions[t]:float(), target[t]):clone()
            
              -- Jordan Neural Network 
              if t == target:size(1) then
                  assert(dlstm_h_out[t] == nil)
                  dlstm_h_out[t] = clones.softmax[t]:backward(lstm_h_out[t], doutput_t):clone()
              else
                  local temp = dTempSummary[t+1]:narrow(1,1,targetDictionarySize)
                  if t ~= 1 then
                    doutput_t:add(clones.exp[t]:backward(predictions[t-1], temp))
                  end
                  dlstm_h_out[t]:add(clones.softmax[t]:backward(lstm_h_out[t], doutput_t))
              end
      
              -- backprop through LSTM timestep
              
              dTempSummary[t], dlstm_c_out[t-1], dlstm_h_out[t-1] = unpack(clones.lstm_out[t]:backward(
                  {inputLSTM_Out[t], lstm_c_out[t-1], lstm_h_out[t-1]},
                  {dlstm_c_out[t], dlstm_h_out[t]}
              ))
              
              if t == target:size(1) then
                dSummary = dTempSummary[t]:narrow(1,targetDictionarySize+1,options.hiddenSize):clone()
              else
                dSummary:add(dTempSummary[t]:narrow(1,targetDictionarySize+1,options.hiddenSize))
              end 
          end
      
          for t=input:size(1),1,-1 do
              -- the error from higher layer is sent only to the last hidden layer
              if t == input:size(1) then
                  assert(dlstm_h[t] == nil)
                  dlstm_h[t] = dSummary:clone()
              end
      
              -- backprop through LSTM timestep
              dembeddings[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t]:backward(
                  {embeddings[t], lstm_c[t-1], lstm_h[t-1]},
                  {dlstm_c[t], dlstm_h[t]}
              ))
              
              -- backprop through embeddings
              clones.embed[t]:backward(input[t]:float(), dembeddings[t])
          end
      
          -- clip gradient element-wise
          grad_params:clamp(-2, 2)
      
          return loss, grad_params
      end
      
      optimMethod(feval, params, optimState)
    end
    
    local cost = 0.0
    cost = validation(validFileName)
    
    print("Elapsed time: " .. os.clock()-nClock)
    
    if epoch == 0 then
      previousCost = cost
      
      local filename = "output/model.net_"..epoch
      os.execute('mkdir -p ' .. sys.dirname(filename))
      torch.save(filename, clones)
      
      filename = "output/optimState_"..epoch
      torch.save(filename, optimState)
      
      epoch = epoch + 1
    else
      if cost > previousCost then
        local filename = "output/model.net_"..(epoch-1)
        clones = torch.load(filename)
        
        optimState.learningRate = optimState.learningRate * 0.7
      else
        previousCost = cost
        
        local filename = "output/model.net_"..epoch
        os.execute('mkdir -p ' .. sys.dirname(filename))
        torch.save(filename, clones)
        
        filename = "output/optimState_"..epoch
        torch.save(filename, optimState)
        
        if epoch%10 == 0 then
          test(options.test, epoch)
        end
    
        epoch = epoch + 1
      end
    end
    
  end
end

function validation(testFileName)

  local cost = 0.0
  local lineCount = 0;
    
  for line in io.lines(testFileName) do
    -- extract every word --
    local sourceWord;
    local targetWord;
    
    lineCount = lineCount + 1
    
    local splitWord = split(line,"\t")
    for index,word in ipairs(splitWord) do
      if index == 1 then
        sourceWord = word
      else
        targetWord = word
      end
    end
  
    local sourceLength = 0
    for word in string.gmatch(sourceWord,"[^ ]+") do
      sourceLength = sourceLength +1 
    end
    sourceLength = sourceLength +1
    
    local targetLength = 0
    for word in string.gmatch(targetWord,"[^ ]+") do
      targetLength = targetLength +1 
    end
    targetLength = targetLength +1
    
    local input= torch.Tensor(sourceLength,1):zero()
    local target = torch.Tensor(targetLength):zero()
  
    --read source and target sequence 
    local currentLength = 0;
    local index = 1;
    for word in string.gmatch(sourceWord,"[^ ]+") do
      if(sourceDictionary[word] == nil) then
        print(word.." Not found in source vocabulary")
      else
        index = sourceDictionary[word]
        input[currentLength + 1][1] = index
      end
      currentLength = currentLength +1; 
    end
    
    index = sourceDictionary["</S>"]
    input[currentLength + 1][1] = index 
    
    currentLength = 0;
    index = 1;
    for word in string.gmatch(targetWord,"[^ ]+") do
      if(targetDictionary[word] == nil) then
        print(word.." Not found in target vocabulary")
      else
        index = targetDictionary[word]
        target[currentLength + 1] = index
      end
      currentLength = currentLength +1; 
    end
  
    index = targetDictionary["</S>"]
    target[currentLength + 1] = index 
    
      local embeddings = {}            -- input embeddings
      local lstm_c = {[0]=initstate_c} -- internal cell states of Encoder LSTM
      local lstm_h = {[0]=initstate_h} -- output values of Encoder LSTM
      
      local lstm_c_out = {[0]=initstate_c_out} -- internal cell states of Decoder LSTM1
      local lstm_h_out = {[0]=initstate_h_out} -- output values of Decoder LSTM1
      
      local predictions = {}           -- softmax outputs
      local inputLSTM_Out = {}
      
      -- Send the input sequence through an LSTM Encoder
      for t=1,input:size(1) do
         embeddings[t] = clones.embed[t]:forward(input[t])[1]
         lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward({embeddings[t], lstm_c[t-1], lstm_h[t-1]}))
      end
      
      -- Last Hidden State of Encoder is the summary of the Source Sequence 
      local summary = lstm_h[input:size(1)]:clone()
      summary = summary:reshape(summary:size(1))
      
      -- Use a Decoder for Prediction target sequence one by one
      for t=1,target:size(1) do
        -- Previous prediction is zero for time t = 1
        if t == 1 then
          inputLSTM_Out[t] = torch.cat(torch.zeros(targetDictionarySize), summary)
          lstm_c_out[t], lstm_h_out[t] = unpack(clones.lstm_out[t]:forward({inputLSTM_Out[t], lstm_c_out[t-1], lstm_h_out[t-1]}))
        else 
          local temp = clones.exp[t]:forward(predictions[t-1]):clone()
          inputLSTM_Out[t] = torch.cat(temp, summary)
          lstm_c_out[t], lstm_h_out[t] = unpack(clones.lstm_out[t]:forward({inputLSTM_Out[t], lstm_c_out[t-1], lstm_h_out[t-1]}))
        end
      
        predictions[t] = clones.softmax[t]:forward(lstm_h_out[t])
        
        cost = cost + clones.criterion[t]:forward(predictions[t]:float(), target[t])
      end
    end
    print("Cost on Development set "..(cost/lineCount))
    return (cost/lineCount)
end

if options.predict then 
  print("On Prediction Mode")

  local filename = "output/model.net_10"
  clones = torch.load(filename)
  
--  clones = {}
--  for name,proto in pairs(protos) do
--      print('cloning '..name)
--      clones[name] = model_utils.clone_many_times(proto, math.max(targetMaxLength,sourceMaxLength)+1, not proto.parameters)
--  end
  
  filename = "output/optimState_10"
  optimState = torch.load(filename)
  
  validation(options.tune)
  test(options.test,-1)
else
  validation(options.tune)
  train(options.train, options.tune)
end
