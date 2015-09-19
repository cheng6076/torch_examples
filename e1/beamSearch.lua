require 'nn'
local LSTM = require 'LSTM'             -- LSTM timestep and utilities
local LSTM_Decoder = require 'LSTM_Decoder'             -- LSTM timestep and utilities
local model_utils=require 'model_utils'
require 'torch'
local class = require 'class'
require 'torch'

  local Hypothesis = class("Hypothesis")
  
  function Hypothesis:__init()
    self.targetOutput = {}
    self.probabilityScore = 0.0
    self.currentCost = 0.0
    self.current_c_out = torch.Tensor(1):zero()
    self.current_h_out = torch.Tensor(1):zero()
    self.done = false
    self.output = {}
  end
  
  function Hypothesis:createTargetWord(targetWord)
    table.insert(self.targetOutput, targetWord)
  end
  
  function Hypothesis:copyTargetWord(targetWord)
    for i = 1,#targetWord do
      table.insert(self.targetOutput, targetWord[i])
    end
  end
  
  function Hypothesis:setTargetWord(targetWord)
    self.output[1] = targetWord
  end
  
  function Hypothesis:getTargetIndex()
    return self.output[1]
  end
  
  function Hypothesis:getTarget()
    return self.targetOutput
  end
  
  function Hypothesis:insertProbability(probability)
    self.probabilityScore = probability
  end
  
  function Hypothesis:getProbability()
    return self.probabilityScore
  end
  
  function Hypothesis:insertCost(cost, previousStateCost)
    self.currentCost = previousStateCost + cost
  end
  
  function Hypothesis:getCost()
    return self.currentCost 
  end
  
  function Hypothesis:insertStateActivations(c_out, h_out)
    self.current_c_out = c_out:clone()
    self.current_h_out = h_out:clone()
  end
  
  function Hypothesis:getStateActivations()
    return self.current_c_out:clone(), self.current_h_out:clone()
  end
  
  function Hypothesis:setDone(Done)
    self.done = Done
  end

function doBeamSearch(summaryState, clones, options, targetDictionarySize, EOS, reverseTargetDictionary)
  local initstate_c_out = torch.zeros( options.hiddenSize)
  local initstate_h_out = initstate_c_out:clone()
  
  -- Initially the predicted target word is nil
  local init_predictions = torch.zeros(targetDictionarySize)
  
  -- concatenate the prediction with the summary obtained
  local temp = init_predictions:clone()
  temp = torch.cat(temp, summaryState:float())
  
  -- obtain the first hidden states
  local lstm_c_out, lstm_h_out = unpack(clones.lstm_out[1]:forward{temp, initstate_c_out, initstate_h_out})
  
  --get logSoftMax Predictions
  local predictions = clones.exp[1]:forward(clones.softmax[1]:forward(lstm_h_out):clone())
  
  -- Use every output as first prediction to build the next set of predictions
  local setOfHypothesis = {}
  for i=1,targetDictionarySize do
    -- Create a initial hypothesis for every target word
    local tempHypothesis = Hypothesis.new()
    -- target word is identified by it's index
    tempHypothesis:createTargetWord(i)
    tempHypothesis:setTargetWord(i)
    tempHypothesis:insertProbability(math.log(predictions[i]))
    tempHypothesis:insertCost(math.log(predictions[i]),0.0)
    tempHypothesis:insertStateActivations(lstm_c_out, lstm_h_out)
    
    if i == EOS then
      tempHypothesis:setDone(true)
    else
      tempHypothesis:setDone(false)
    end
    
    -- insert the new hypothesis into the set of base hypothesis
    table.insert(setOfHypothesis,tempHypothesis)
  end
  
--  for iterate =1, #setOfHypothesis do
--      local hypothesisPairs = setOfHypothesis[iterate]
--      print("Initial Hypothesis "..iterate.." out of "..(#setOfHypothesis))
--      print(hypothesisPairs:getProbability())
--      local temptarget = hypothesisPairs:getTarget()
--      for j = 1,#temptarget do
--        print(reverseTargetDictionary[temptarget[j]])
--      end
--  end
  
  local newHypothesis = {}
  local currentTableLength = 0
   
  local flag = true
  local count = 1
    
  while(flag and count < 100) do
  -- no hypothesis generated initially
    newHypothesis = {}
    count = count +1
        
    if count > 100 then
      flag = false
      break
    end
    
    local completedHypothesis = 0  
    
    -- for every hypothesis in the base set of hypothesis
    for i =1, #setOfHypothesis do
      local hypothesisPairs = setOfHypothesis[i]
      -- get the state of the network and the previous word generated
      local prev_c_out, prev_h_out = hypothesisPairs:getStateActivations()
      local getPrevIndex = hypothesisPairs:getTargetIndex()
--      print(getPrevIndex)
      
      -- If the hypothesis has already generated EOS then add it to the new set of hypothesis
      if getPrevIndex == EOS then
        completedHypothesis = completedHypothesis +1
        
        local newHypothesisTemp = Hypothesis.new()
        newHypothesisTemp:copyTargetWord(hypothesisPairs:getTarget())
        newHypothesisTemp:setTargetWord(getPrevIndex)
        newHypothesisTemp:insertProbability(hypothesisPairs:getProbability())
        newHypothesisTemp:insertCost(hypothesisPairs:getCost(),0.0)
        newHypothesisTemp:insertStateActivations(prev_c_out, prev_h_out)
        
        table.insert(newHypothesis, newHypothesisTemp)
        
        if completedHypothesis == #setOfHypothesis then
          flag = false
          break
        end
      else
        -- The previous hypothesis generated non-EOS so generate next token based on this 
        local temp = torch.zeros(targetDictionarySize)
        temp[getPrevIndex] = 1.0
        temp = torch.cat(temp, summaryState:float())
        
        local lstm_c_out_new, lstm_h_out_new = unpack(clones.lstm_out[1]:forward{temp, prev_c_out, prev_h_out})
        local predictions_new = clones.exp[1]:forward(clones.softmax[1]:forward(lstm_h_out_new):clone())
        
        -- For every target word generate a new hypothesis
        for j=1,targetDictionarySize do
          -- add the list of already generated words          
          local tempHypothesis = Hypothesis.new()
          t = hypothesisPairs:getTarget()
          for ii = 1,#t do
            tempHypothesis:createTargetWord(t[ii])
          end
          
          tempHypothesis:createTargetWord(j)
          tempHypothesis:setTargetWord(j)
          
          tempHypothesis:insertProbability(math.log(predictions_new[j]))
          tempHypothesis:insertCost(math.log(predictions_new[j]), hypothesisPairs:getCost())
          tempHypothesis:insertStateActivations(lstm_c_out_new, lstm_h_out_new)
          
          if j == EOS then
            tempHypothesis:setDone(true)
          else
            tempHypothesis:setDone(false)
          end
          
          table.insert(newHypothesis,tempHypothesis)
        end
      end
    end
    
    -- Need to prune out hypothesis based on beams size
    -- sort the hypothesis in descending order of the log-probability values
    table.sort(newHypothesis, function(hypo1, hypo2) return hypo1:getCost() > hypo2:getCost() end)
    
    local currentTableLength = 0
    setOfHypothesis = {}
    
    while currentTableLength < 5 and currentTableLength < #newHypothesis do
      local newHypothesisTemp = Hypothesis.new()
      newHypothesisTemp:copyTargetWord(newHypothesis[currentTableLength +1]:getTarget())
      newHypothesisTemp:setTargetWord(newHypothesis[currentTableLength +1]:getTargetIndex())
      newHypothesisTemp:insertProbability(newHypothesis[currentTableLength +1]:getProbability())
      newHypothesisTemp:insertCost(newHypothesis[currentTableLength +1]:getCost(),0.0)
      
      local c_out, h_out = newHypothesis[currentTableLength +1]:getStateActivations()
      newHypothesisTemp:insertStateActivations(c_out, h_out)
      
      table.insert(setOfHypothesis, newHypothesisTemp)
      currentTableLength = currentTableLength + 1  
    end
    
--    for iterate =1, #setOfHypothesis do
--      local hypothesisPairs = setOfHypothesis[iterate]
--      print("Initial Hypothesis "..iterate.." out of "..(#setOfHypothesis))
--      print(hypothesisPairs:getProbability())
--      local temptarget = hypothesisPairs:getTarget()
--      for j = 1,#temptarget do
--        print(reverseTargetDictionary[temptarget[j]])
--      end
--    end
    
    newHypothesis = {}
  end
  
  local predictedSequence = {}
  predictedSequence = setOfHypothesis[1]:getTarget()
  
  return predictedSequence, setOfHypothesis[1]:getCost()
  
end