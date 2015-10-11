-- neuralsum

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.Squeeze'
require 'util.misc'
BatchLoader = require 'util.b'
model_utils = require 'util.model_utils'

local stringx = require('pl.stringx')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/ptb','data directory. Should contain train.txt/valid.txt/test.txt with input data')
-- model params
cmd:option('-rnn_size', 150, 'size of LSTM internal state, dimentionality of document embedding')
cmd:option('-sentence_vec_size', 150, 'dimensionality of sentence embeddings')
cmd:option('-word_vec_size', 15, 'dimensionality of word embeddings')
cmd:option('-feature_maps', '{50,100,150,200,200,200,200}', 'number of feature maps in the CNN')
cmd:option('-kernels', '{1,2,3,4,5,6,7}', 'conv net kernel widths')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-dropout',0.5,'dropout. 0 = no dropout')
-- optimization
cmd:option('-learning_rate',1,'starting learning rate')
cmd:option('-learning_rate_decay',0.5,'learning rate decay')
cmd:option('-decay_when',1,'decay if validation perplexity does not improve by more than this much')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-batch_norm', 0, 'use batch normalization over input embeddings (1=yes)')
cmd:option('-batch_size',20,'number of sequences to train on in parallel')
cmd:option('-max_epochs',25,'number of full passes through the training data')
cmd:option('-max_grad_norm',5,'normalize gradients at')
cmd:option('-max_sentence_l',20,'maximum sentence length')
cmd:option('-max_document_l',10,'maximum document length')
cmd:option('-threads', 16, 'number of threads') 
-- bookkeeping
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-print_every',5,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every', 1, 'save every n epochs')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','word','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-checkpoint', 'checkpoint.t7', 'start from a checkpoint if a valid checkpoint.t7 file is given')
cmd:option('-EOS', '', '<EOS> symbol. should be a single unused word (like +) for PTB and blank for others')
-- GPU/CPU
cmd:option('-gpuid', -1,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 0,'use cudnn (1=yes). this should greatly speed up convolutions')
cmd:option('-time', 0, 'print batch times')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)


--if opt.threads > 0 then
--    torch.setnumthreads(opt.threads)
--end

-- some housekeeping
loadstring('opt.kernels = ' .. opt.kernels)() -- get kernel sizes
loadstring('opt.feature_maps = ' .. opt.feature_maps)() -- get feature map sizes

opt.padding = 0 

-- global constants for certain tokens
opt.tokens = {}
opt.tokens.EOS = opt.EOS
opt.tokens.UNK = '|' -- unk word token
opt.tokens.START = '{' -- start-of-word token
opt.tokens.END = '}' -- end-of-word token
opt.tokens.ZEROPAD = ' ' -- zero-pad token 

-- load necessary packages depending on config options
if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1)
end

if opt.cudnn == 1 then
   assert(opt.gpuid >= 0, 'GPU must be used if using cudnn')
   print('using cudnn...')
   require 'cudnn'
end

-- create the data loader class
loader = BatchLoader.create(opt.batch_size, opt.padding, opt.max_sentence_l, opt.max_document_l)
print('Word vocab size: ' .. #loader.idx2word)
opt.max_sentence_l = loader.max_sentence_l
opt.max_document_l = loader.max_document_l
opt.seq_length = opt.max_document_l
--print (opt.max_sentence_l)

-- load model objects. we do this here because of cudnn options
TDNN = require 'model.TDNN'
LSTMTDNN = require 'model.LSTMTDNN'

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

if path.exists(opt.checkpoint) then -- start re-training from a checkpoint
   print('loading ' .. opt.checkpoint .. ' for retraining')
   checkpoint = torch.load(opt.checkpoint)
   opt = checkpoint.opt
   retrain = true
end

-- define the model: prototypes for one timestep, then clone them in time
protos = {}
print('creating an LSTM-CNN with ' .. opt.num_layers .. ' layers')
if retrain then
    protos = checkpoint.protos
else
    protos.rnn = LSTMTDNN.lstmtdnn(opt.rnn_size, opt.num_layers, opt.dropout,
				opt.sentence_vec_size, #loader.idx2word, opt.word_vec_size, opt.feature_maps,
				opt.kernels, loader.max_sentence_l, opt.batch_norm)
    print (opt.rnn_size)    
    local d = nn.Identity()()
    local s = nn.Identity()()
    local score = nn.CAddTable()({nn.Linear(opt.rnn_size, 2)(d),nn.Linear(opt.rnn_size, 2)(s)})
    local logsoft = nn.LogSoftMax()(score)
    protos.extractor = nn.gModule({d,s},{logsoft})
    -- training criterion (negative log likelihood)
    protos.criterion = nn.ClassNLLCriterion()
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
print('number of parameters in the model: ' .. params:nElement())

-- initialization
if not retrain then
   params:uniform(-opt.param_init, opt.param_init) -- small numbers uniform if starting from scratch
end


-- get layers which will be referenced layer (during SGD or introspection)
function get_layer(layer)
    local tn = torch.typename(layer)
    if layer.name ~= nil then
	if layer.name == 'word_vecs' then
	    word_vecs = layer
	elseif layer.name == 'cnn' then
	    cnn = layer
	end
    end
end 
protos.rnn:apply(get_layer)

-- make a bunch of clones after flattening, as that reallocates memory
-- perhaps this is because we want to keep the output at every timestep?
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- this returns words at time t, previous c, previous h
function get_input(x_word, t, prev_states)
    local u = {}
    table.insert(u, x_word[{{},t}])
    for i = 1, #prev_states do table.insert(u, prev_states[i]) end
    return u
end

-- evaluate the loss over an entire split
function eval_split(split_idx, max_batches)
    print('evaluating loss over split index ' .. split_idx)
    local n = loader.split_sizes[split_idx]

    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_idx) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}    
    if split_idx<=2 then -- batch eval        
	for i = 1,n do -- iterate over batches in the split
	    -- fetch a batch
	    local x_word, y = loader:next_batch(split_idx)
	    if opt.gpuid >= 0 then -- ship the input arrays to GPU
		-- have to convert to float because integers can't be cuda()'d
		x_word = x_word:float():cuda()
		y = y:float():cuda()
	    end
	    -- forward pass
	    for t=1,opt.seq_length do
		clones.rnn[t]:evaluate() -- for dropout proper functioning
		local lst = clones.rnn[t]:forward(get_input(x_word, t, rnn_state[t-1]))
		rnn_state[t] = {}
		        for i=1,#init_state do
                    table.insert(rnn_state[t], lst[i])
                end
	    end
            local dvec = rnn_state[opt.seq_length][#init_state]

            for t=1,opt.seq_length do
                prediction = clones.extractor[t]:forward({dvec, rnn_state[t][#init_state]})  --the last state is h
                loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}])
            end
             

	end
	loss = loss / opt.seq_length / n
    else -- full eval on test set
        local x_word, y = loader:next_batch(split_idx)
	if opt.gpuid >= 0 then -- ship the input arrays to GPU
	    -- have to convert to float because integers can't be cuda()'d
	    y = y:float():cuda()
	    x_word = x_word:float():cuda()
	end
	protos.rnn:evaluate() -- just need one clone

	for t = 1, x:size(2) do
	    local lst = protos.rnn:forward(get_input(x_word, t, rnn_state[t-1]))
	    rnn_state[t] = {}
	    for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
    end
    local dvec = rnn_state[x:size(2)][#init_state]
    for t = 1, x:size(2) do
        prediction = protos.extractor:forward({dvec, rnn_state[t][#init_state]})
        local tok_perp
        tok_perp = protos.criterion:forward(prediction, y[{{},t}])
        loss = loss + tok_perp
	end
	loss = loss / x:size(2)
    end    
    local perp = torch.exp(loss)    
    return perp
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)

function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    ------------------ get minibatch -------------------

    local x_word, y = loader:next_batch(1) --from train

    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x_word = x:float():cuda()
        y = y:float():cuda()
    end
    ------------------- forward pass -------------------

    local rnn_state = {[0] = init_state_global}
    local hvecs = {}
    local predictions = {}           -- softmax outputs
    local loss = 0

    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)        
        local lst = clones.rnn[t]:forward(get_input(x_word, t, rnn_state[t-1]))
        rnn_state[t] = {}
        for i=1,#init_state do
            table.insert(rnn_state[t], lst[i])
        end -- extract the state

        table.insert(hvecs, rnn_state[t][#init_state])
    end

    local document = rnn_state[opt.seq_length][#init_state]:clone() -- last h

    for t=1,opt.seq_length do
        -- is clones really necessary here?
        predictions[t] = clones.extractor[t]:forward({document, hvecs[t]})
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end

    loss = loss / opt.seq_length

    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)

    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones

    for t=opt.seq_length,1,-1 do
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        local dhvec_t = clones.extractor[t]:backward(hvecs[t], doutput_t) -- this will create two values
        drnn_state[t][#init_state]:add(dhvec_t[1]:add(dhvec_t[2]))
        local dlst = clones.rnn[t]:backward(get_input(x_word, t, rnn_state[t-1]), drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end	
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state]
    
    -- renormalize gradients
    local grad_norm, shrink_factor
    grad_norm = grad_params:norm()
    if grad_norm > opt.max_grad_norm then
        shrink_factor = opt.max_grad_norm / grad_norm
        grad_params:mul(shrink_factor)
    end    
    params:add(grad_params:mul(-lr)) -- update params
    return loss
end


-- start optimization here
train_losses = {}
val_losses = {}
lr = opt.learning_rate -- starting learning rate which will be decayed
local iterations = opt.max_epochs * loader.split_sizes[1]
if wor_vecs ~= nil then word_vecs.weight[1]:zero() end -- zero-padding vector is always zero
for i = 1, iterations do

    local epoch = i / loader.split_sizes[1]
    local timer = torch.Timer()
    local time = timer:time().real
    train_loss = feval(params) -- fwd/backprop and update params
    if word_vecs ~= nil then -- zero-padding vector is always zero
        word_vecs.weight[1]:zero()
        word_vecs.gradWeight[1]:zero()
    end 
    train_losses[i] = train_loss

    -- every now and then or on last iteration
    if i % loader.split_sizes[1] == 0 then
        print ('evaluate')
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[#val_losses+1] = val_loss
        local savefile = string.format('%s/lm_%s_epoch%.2f_%.2f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = {loader.idx2word, loader.word2idx}
	    checkpoint.lr = lr
        print('saving checkpoint to ' .. savefile)
        if epoch == opt.max_epochs or epoch % opt.save_every == 0 then
            torch.save(savefile, checkpoint)
        end
    end

    -- decay learning rate after epoch
    if i % loader.split_sizes[1] == 0 and #val_losses > 2 then
        if val_losses[#val_losses-1] - val_losses[#val_losses] < opt.decay_when then
            lr = lr * opt.learning_rate_decay
	end
    end    

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.2f), train_loss = %6.4f", i, iterations, epoch, train_loss))
    end   
    if i % 10 == 0 then collectgarbage() end
    if opt.time ~= 0 then
       print("Batch Time:", timer:time().real - time)
    end
end

--evaluate on full test set.

test_result = eval_split(3)
print('test set: ' .. test_result)

