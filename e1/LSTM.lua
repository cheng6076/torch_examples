-- adapted from: wojciechz/learning_to_execute on github
require 'nngraph'
nngraph.setDebug(true)

local LSTM = {}

-- Creates one timestep of one LSTM
function LSTM.lstm(inputSize, hiddenSize)
    local inputs = {}
    table.insert(inputs, nn.Identity()())   -- network input
    table.insert(inputs, nn.Identity()())   -- c at time t-1
    table.insert(inputs, nn.Identity()())   -- h at time t-1
    local input = inputs[1]
    local prev_c = inputs[2]
    local prev_h = inputs[3]
    
    local i2h = nn.Linear(inputSize, 4 * hiddenSize)(input)  -- input to hidden
    local h2h = nn.Linear(hiddenSize, 4 * hiddenSize)(prev_h)   -- hidden to hidden
    local preactivations = nn.CAddTable()({i2h, h2h})       -- i2h + h2h
    
    -- gates
    local pre_sigmoid_chunk = nn.Narrow(1, 1, 3 * hiddenSize)(preactivations)
    local all_gates = nn.Sigmoid()(pre_sigmoid_chunk)
    
    -- input
    local in_chunk = nn.Narrow(1, 3 * hiddenSize + 1, hiddenSize)(preactivations)
    local in_transform = nn.Tanh()(in_chunk)
    
    local in_gate = nn.Narrow(1, 1, hiddenSize)(all_gates)
    local forget_gate = nn.Narrow(1, hiddenSize + 1, hiddenSize)(all_gates)
    local out_gate = nn.Narrow(1, 2 * hiddenSize + 1, hiddenSize)(all_gates)
    
    -- previous cell state contribution
    local c_forget = nn.CMulTable()({forget_gate, prev_c})
    -- input contribution
    local c_input = nn.CMulTable()({in_gate, in_transform})
    -- next cell state
    local next_c = nn.CAddTable()({
      c_forget,
      c_input
    })

    local c_transform = nn.Tanh()(next_c)
    local next_h = nn.CMulTable()({out_gate, c_transform})
  
    -- module outputs
    local outputs = {}
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
    
    -- packs the graph into a convenient module with standard API (:forward(), :backward())
    return nn.gModule(inputs, outputs)
end

return LSTM

