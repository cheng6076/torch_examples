--[[
model introspection
--]]

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.Squeeze'
require 'util.misc'

BatchLoader = require 'util.BatchLoaderUnk'
model_utils = require 'util.model_utils'

HighwayMLP = require 'model.HighwayMLP'
TDNN = require 'model.TDNN'
LSTMTDNN = require 'model.LSTMTDNN'

local stringx = require('pl.stringx')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Perform model introspection')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-model','final-results/en-large-word-model.t7', 'model file')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-savefile', 'chargrams.tsv', 'save max chargrams to')
cmd:text()

-- parse input params
opt2 = cmd:parse(arg)

if opt2.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt2.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt2.gpuid + 1)
end


-- load model
checkpoint = torch.load(opt2.model)
opt = checkpoint.opt
torch.manualSeed(opt.seed)
protos = checkpoint.protos
idx2word, word2idx, idx2char, char2idx = table.unpack(checkpoint.vocab)


function word2char2idx(word)
    local char_idx = torch.zeros(opt.max_word_l)
    char_idx:fill(1) -- fill with padding first
    local l = opt.padding + 1 -- skip beginning padding
    for c in word:gmatch'.' do
        -- while character is valid and we are under max word length
        if char2idx[c] ~= nil and l <= char_idx:size(1) then
	    char_idx[l] = char2idx[c]
	    l = l + 1
	end
    end
    return char_idx
end

-- get layers which will be referenced layer (during SGD or introspection)
function get_layer(layer)
    local tn = torch.typename(layer)
    if layer.name ~= nil then
        if layer.name == 'word_vecs' then
	    word_vecs = layer
	elseif layer.name == 'char_vecs' then
	    char_vecs = layer
	elseif layer.name == 'cnn' then
	    cnn = layer
	elseif layer.name == 'highway' then
	    highway = layer
	end
    end
end 
protos.rnn:apply(get_layer)

-- get conv filter layers
if cnn ~= nil then
   conv_filters = {}
   cnn:apply(function (x) if x.name ~= nil then if x.name:sub(1,4)=='conv' then 
			     table.insert(conv_filters, x) end end end)
end
-- for each word get the feature map values as well
-- as the chargrams that activate the feature map (i.e. max)
function get_max_chargrams()
    local result = {}
    local char_idx_all = torch.zeros(#idx2word, opt.max_word_l)
    for i = 1, #idx2word do
        char_idx_all[i] = word2char2idx(opt.tokens.START .. idx2word[i] .. opt.tokens.END)
    end
    local char_vecs_all = char_vecs:forward(char_idx_all) -- vocab_size x max_word_l x char_vec_size
    for i = 1, #conv_filters do
    	local max_val, max_arg
        local conv_filter = conv_filters[i]
	local width = conv_filter.kW
	result[width] = {}
	local conv_output = conv_filter:forward(char_vecs_all)
	max_val, max_arg = torch.max(conv_output,2) -- get max values and argmaxes
	max_val = max_val:squeeze()
	max_arg =  max_arg:squeeze()
	result[width][1] = max_val
	result[width][2] = {} -- this is where we'll store the chargrams (as strings)
	for j = 1, #idx2word do
    	    local chargrams = {}
	    for k = 1, max_arg:size(2) do
	        local c = {}
	        local start_char = max_arg[j][k] 
		local end_char = max_arg[j][k] + width - 1
		for l = start_char, end_char do
		    table.insert(c, idx2char[char_idx_all[j][l]])
		end
		chargrams[#chargrams + 1] = table.concat(c)
	    end
	    result[width][2][j] = chargrams
	end
    end
    return result
end

-- get max chargrams and print them out to file
function print_max_chargrams()
    result = get_max_chargrams()
    max_chargrams = {}
    for u,v in pairs(result) do
	local max_val, max_arg = torch.max(v[1],1)
	max_val = max_val:squeeze()
	max_arg = max_arg:squeeze()
	for i = 1, max_arg:size(1) do 
	    local chargram = v[2][max_arg[i]][i]
	    local word = idx2word[max_arg[i]]
	    max_chargrams[#max_chargrams + 1] = {u, chargram, word, max_val[i]}
	end
    end

    local f = io.open(opt2.savefile, 'w')
    for u,v in ipairs(max_chargrams) do
	f:write(v[1]..'\t'..v[2]..'\t'..v[3]..'\t'..v[4]..'\n')
    end
    f:close()
end

-- get all character ngrams that occur in the corpus, and represent them as vectors
function get_all_chargrams(idx2word)
    local idx2chargram = {}
    local chargram2idx = {}
    local count = {}
    for i = 1, #idx2word do
        local ngrams = get_chargrams(opt.tokens.START .. idx2word[i] .. opt.tokens.END, 2, 7)
	for _, ngram in pairs(ngrams) do
	    if count[ngram] == nil then
	        count[ngram] = 1
	    else
	        count[ngram] = count[ngram] + 1	     
	    end
	end
    end
    for ngram, c in pairs(count) do
        if c > 3 then
	    idx2chargram[#idx2chargram + 1] = ngram
	    chargram2idx[ngram] = #idx2chargram	    
	end
    end
    return idx2chargram, chargram2idx
end

function get_chargrams(word, low, high)
    local ngrams = {}
    local word_len = word:len()
    local high = math.min(high, word_len)
    for i = low, high do
        get_chargram(word, word_len, i, ngrams)
    end
    return ngrams
end

function get_chargram(word, word_len, n, ngrams)
    for i = 1, word_len-n+1 do
        local ngram = word:sub(i, i+n-1)
	ngrams[#ngrams + 1] = ngram
    end
end

function get_chargram_vecs(savefile)
    idx2chargram, chargram2idx = get_all_chargrams(idx2word)
    print(#idx2chargram)
    chargram_idx_all = torch.zeros(#idx2chargram, opt.max_word_l)
    for i = 1, #idx2chargram do
	chargram_idx_all[i] = word2char2idx(idx2chargram[i], opt.max_word_l)
    end
    chargram_vecs_all = char_vecs:forward(chargram_idx_all)
    chargram_vecs = torch.zeros(#idx2chargram, torch.sum(torch.Tensor(opt.feature_maps)))
    for i = 1, #idx2chargram do
        chargram_vecs[i] = cnn:forward(chargram_vecs_all[i]:view(1, opt.max_word_l, opt.char_vec_size)):float()
    end
    local f = io.open(savefile..'-dic.txt', 'w')
    for _, ngram in ipairs(idx2chargram) do
        f:write(ngram..'\n')
    end
    f:close()
    torch.save(savefile..'.t7',  chargram_vecs)
end

--get contribution of each character to the feature vector by counting
function get_contribution()
    result ={}
    result2 = {}
    local char_idx_all = torch.zeros(#idx2word, opt.max_word_l)
    for i = 1, #idx2word do
        char_idx_all[i] = word2char2idx(opt.tokens.START .. idx2word[i] .. opt.tokens.END)
	result2[i] = {}
    end
    local result = torch.zeros(char_idx_all:size())
    local char_vecs_all = char_vecs:forward(char_idx_all)
    for i = 1, #conv_filters do
        local conv_filter = conv_filters[i]
	local width = conv_filter.kW
	local conv_output = conv_filter:forward(char_vecs_all)
	max_val, max_arg = torch.max(conv_output, 2)
	max_val = max_val:squeeze()
	max_arg = max_arg:squeeze()
	for j = 1, #idx2word do
	    local chargrams = {}
	    for k = 1, max_arg:size(2) do
	        local c = {}
		local start_char = max_arg[j][k]
		local end_char = max_arg[j][k] + width - 1
		for l = start_char, end_char do	
		    result[j][l] = result[j][l] + 1
		    table.insert(c, idx2char[char_idx_all[j][l]])
		end
		if result2[j][c] == nil then
		    result2[j][c] = 1
		else
		    result2[j][c] = result2[j][c] + 1
		end
	    end
	end
    end
end

function get_nn(words, k)
    local k = k or 5
    if opt.use_chars==1 then
        word_vecs_idx = torch.zeros(#idx2word, opt.max_word_l)
        for i = 1, #idx2word do
	    word_vecs_idx[i] = word2char2idx(opt.tokens.START .. idx2word[i] .. opt.tokens.END, opt.max_word_l)
        end
	if opt.gpuid >= 0 then
	    word_vecs_idx = word_vecs_idx:float():cuda()
	end
	if word_vecs_trained == nil then
	    word_vecs_trained = cnn:forward(char_vecs:forward(word_vecs_idx))
	    word_vecs_trained = highway:forward(word_vecs_trained)
	end
	collectgarbage()
    else
        word_vecs_trained = word_vecs.weight
    end
    -- normalize 
    word_vecs_trained = word_vecs_trained:float()
    word_vecs_trained = normalize(word_vecs_trained)
    for i = 1, #words do
        local word = words[i]
	if word2idx[word] == nil then
	    new_word = torch.zeros(2, opt.max_word_l)
	    new_word[1] = word2char2idx(opt.tokens.START .. word .. opt.tokens.END, opt.max_word_l)
	    new_word[2] = torch.ones(opt.max_word_l)
	    if opt.gpuid >= 0 then
	        new_word = new_word:float():cuda()
	    end
	    new_word = cnn:forward(char_vecs:forward(new_word))
	    new_word = highway:forward(new_word)
	    new_word = new_word[1] / torch.norm(new_word[1])
	    new_word = new_word:double()
	    print('----new word----')
        else
	    new_word = word_vecs_trained[word2idx[word]]
	end
        r = get_sim_words(word_vecs_trained, new_word, k)	
        print('----'..word..'----')
    	for j = 1, k do
	    print(string.format('%s, %.4f', r[j][1], r[j][2]))
	end
    end       
end

function normalize(m)
    local m_norm = torch.zeros(m:size())    
    for i = 1, m:size(1) do
        m_norm[i] = m[i] / torch.norm(m[i])
    end
    return m_norm
end

function get_sim_words(m, w, k)
    local k = k or 5
    local sim = torch.mv(m, w)
    sim, idx = torch.sort(-sim)
    local r = {}
    for i = 1, k do
        r[i] = {idx2word[idx[i]], -sim[i]}
    end
    return r
end
