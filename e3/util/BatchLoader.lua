
-- Modified from https://github.com/karpathy/char-rnn
-- This version is for cases where one has already segmented train/val/test splits

local BatchLoaderUnk = {}
local stringx = require('pl.stringx')
BatchLoaderUnk.__index = BatchLoaderUnk

function BatchLoaderUnk.create(data_dir, batch_size, seq_length, padding, max_word_l)
    local self = {}
    setmetatable(self, BatchLoaderUnk)
    self.padding = padding or 0
    local train_file = path.join(data_dir, 'train.txt')
    local valid_file = path.join(data_dir, 'valid.txt')
    local test_file = path.join(data_dir, 'test.txt')
    local input_files = {train_file, valid_file, test_file}
    local vocab_file = path.join(data_dir, 'vocab.t7')
    local tensor_file = path.join(data_dir, 'data.t7')
    local char_file = path.join(data_dir, 'data_char.t7')

    -- construct a tensor with all the data
    if not (path.exists(vocab_file) or path.exists(tensor_file) or path.exists(char_file)) then
        print('one-time setup: preprocessing input train/valid/test files in dir: ' .. data_dir)
        BatchLoaderUnk.text_to_tensor(input_files, vocab_file, tensor_file, char_file)
    end

    print('loading data files...')
    local all_data = torch.load(tensor_file) -- train, valid, test tensors
    local all_data_char = torch.load(char_file) -- train, valid, test character indices
    local vocab_mapping = torch.load(vocab_file)
    self.idx2word, self.word2idx, self.idx2char, self.char2idx = table.unpack(vocab_mapping)
    self.vocab_size = #self.idx2word
    print(string.format('Word vocab size: %d, Char vocab size: %d', #self.idx2word, #self.idx2char))
    -- create word-char mappings
    if max_word_l == nil then -- if max word length is not specified
	self.max_word_l = 0
	for i = 1, #self.idx2word do
	    self.max_word_l = math.max(self.max_word_l, self.idx2word[i]:len()) -- get max word length 
	end
    else
        self.max_word_l = max_word_l
    end
    self.max_word_l = self.max_word_l + self.padding
    self.word2char2idx = torch.zeros(#self.idx2word, self.max_word_l + 2*self.padding):long()
    for i = 1, #self.idx2word do
        self.word2char2idx[i] = self:get_word2char2idx("{"..self.idx2word[i].."}")
    end
    -- cut off the end for train/valid sets so that it divides evenly
    -- test set is not cut off
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.split_sizes = {}
    self.all_batches = {}
    print('reshaping tensors...')  
    local x_batches, y_batches, nbatches
    for split, data in ipairs(all_data) do
    	local len = data:size(1)
	if len % (batch_size * seq_length) ~= 0 and split < 3 then
	    data = data:sub(1, batch_size * seq_length * math.floor(len / (batch_size * seq_length)))
	end
	local ydata = data:clone()
	ydata:sub(1,-2):copy(data:sub(2,-1))
	ydata[-1] = data[1]
	local data_char = torch.zeros(data:size(1), self.max_word_l):long()
	for i = 1, data:(1) do
	    data_char[i] = self:expand(all_data_char[split][i])
	end
	if split < 3 then
	    x_batches = data:view(batch_size, -1):split(seq_length, 2)
	    y_batches = ydata:view(batch_size, -1):split(seq_length, 2)
	    nbatches = #x_batches	   
	    self.split_sizes[split] = nbatches
	    assert(#x_batches == #y_batches)
	else --for test we repeat dimensions to batch size (easier but inefficient evaluation)
	    x_batches = {data:resize(1, data:size(1)):expand(batch_size, data:size(2))}
	    y_batches = {ydata:resize(1, ydata:size(1)):expand(batch_size, ydata:size(2))}
	    self.split_sizes[split] = 1	
	end
	local x_char_batches = {}
	for i = 1, #x_batches do
	    x_char_batches[#x_char_batches + 1] = lookup:forward(x_batches[i]):clone()
	end
  	self.all_batches[split] = {x_batches, y_batches, x_char_batches}
    end
    self.batch_idx = {0,0,0}
    print(string.format('data load done. Number of batches in train: %d, val: %d, test: %d', self.split_sizes[1], self.split_sizes[2], self.split_sizes[3]))
    collectgarbage()
    return self
end

function BatchLoaderUnk:get_word2char2idx(word)
    local char_idx = torch.zeros(self.max_word_l+ 2*self.padding)
    char_idx:fill(self.char2idx[' ']) -- fill with padding first
    local l = self.padding + 1
    for c in word:gmatch'.' do
        if self.char2idx[c] ~= nil and l <= char_idx:size(1) then -- cutoff if word is too long
	    char_idx[l] = self.char2idx[c]
	    l = l + 1
	end
    end
    return char_idx:long()
end

function BatchLoaderUnk:expand(t)
    for i = 1, self.padding do
        table.insert(t, 1, 1) -- 1 is always char idx for zero pad
    end
    while #t < self.max_word_l do
        table.insert(t, 1, 1)
    end
    return torch.LongTensor(t):sub(1, self.max_word_l)
end

function BatchLoaderUnk:reset_batch_pointer(split_idx, batch_idx)
    batch_idx = batch_idx or 0
    self.batch_idx[split_idx] = batch_idx
end

function BatchLoaderUnk:next_batch(split_idx)
    -- split_idx is integer: 1 = train, 2 = val, 3 = test
    self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
    if self.batch_idx[split_idx] > self.split_sizes[split_idx] then
        self.batch_idx[split_idx] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local idx = self.batch_idx[split_idx]
    return self.all_batches[split_idx][1][idx], self.all_batches[split_idx][2][idx], self.all_batches[split_idx][3][idx]
end

function BatchLoaderUnk.text_to_tensor(input_files, out_vocabfile, out_tensorfile, out_charfile)
    print('Processing text into tensors...')
    local f, rawdata, output, output_char
    local output_tensors = {} -- output tensors for train/val/test
    local output_chars = {} -- output character for train/val/test sets (not tensors yet)
    local vocab_count = {} -- vocab count 
    local idx2word = {'|'} -- unknown word token
    local word2idx = {}; word2idx['|'] = 1
    local idx2char = {' ','{','}'} -- zero-pad, start-of-word, end-of-word tokens
    local char2idx = {}; char2idx[' '] = 1; char2idx['{'] = 2, char2idx['}'] = 3
    for	split = 1,3 do -- split = 1 (train), 2 (val), or 3 (test)
        output = {}
	output_char = {}
        f = torch.DiskFile(input_files[split])
	rawdata = f:readString('*a') -- read all data at once
	f:close()
	rawdata = stringx.replace(rawdata, '\n', '+') -- use '+' instead of '<eos>' for end-of-sentence
	rawdata = stringx.replace(rawdata, '{', ' ') -- '{' is reserved for start-of-word symbol
	rawdata = stringx.replace(rawdata, '}', ' ') -- '}' is reserved for end-of-word symbol
	rawdata = stringx.replace(rawdata, '<unk>', '|') -- '<unk>' gets replaced with a single character
	for word in rawdata:gmatch'([^%s]+)' do
	    local chars = {char2idx['{']} -- start-of-word symbol
	    if string.sub(word,1,1) == '|' and word:len() > 1 then -- unk token with character info available
	        word = string.sub(word, 3)
		output[#output + 1] = word2idx['|']
	    else
		if word2idx[word]==nil then
		    idx2word[#idx2word + 1] = word -- create word-idx/idx-word mappings
		    word2idx[word] = #idx2word
		end
		output[#output + 1] = word2idx[word]		
	    end
	    for char in word:gmatch'.' do
		if char2idx[char]==nil then
		    idx2char[#idx2char + 1] = char -- create char-idx/idx-char mappings
		    char2idx[char] = #idx2char
		end
		chars[#chars + 1] = char2idx[char]
	    end
	    chars[#chars + 1] = char2idx['}'] -- end-of-word symbol
	end	
	output_tensors[split] = torch.LongTensor(output)
	output_chars[split] = output_char
    end

    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, {idx2word, word2idx, idx2char, char2idx})
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, output_tensors)
    print('saving ' .. out_charfile)
    torch.save(out_charfile, output_chars)
end

return BatchLoaderUnk

