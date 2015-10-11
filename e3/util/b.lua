require 'lfs'
local stringx = require 'pl.stringx'
local BatchLoaderUnk = {}
BatchLoaderUnk.__index = BatchLoaderUnk

function BatchLoaderUnk.create(batch_size, padding, max_sentence_l, max_document_l)
    local self = {}
    setmetatable(self, BatchLoaderUnk)
    self.padding = padding or 0
    local train_file = 'dataset/train/'
    local valid_file = 'dataset/validation/'
    local test_file = 'dataset/test/'
    local input_files = {train_file, valid_file, test_file}
    local vocab_file = 'dataset/vocab.t7'
    local word_file = 'dataset/data_word.t7'
    local label_file = 'dataset/data_label.t7'

    if not (path.exists(vocab_file) or path.exists(word_file) or path.exists(label_file)) then
        print('one-time setup: preprocessing input train/valid/test files')
        BatchLoaderUnk.text_to_tensor(input_files, vocab_file, word_file, label_file, max_sentence_l, max_document_l)
    end

    print('loading data files...')
    local all_data_word = torch.load(word_file)
    local all_label = torch.load(label_file)
    local vocab_mapping = torch.load(vocab_file)
    self.idx2word, self.word2idx = table.unpack(vocab_mapping)
    self.vocab_size = #self.idx2word
    print(string.format('Word vocab size: %d', #self.idx2word))

    self.max_sentence_l = all_data_word[1]:size(3)
    self.max_document_l = all_data_word[1]:size(2)
    self.batch_size = batch_size
    self.split_sizes = {}
    self.all_batches = {}
    print('reshaping tensors...')
    local x_batches, y_batches, nbatches
    for split= 1, 3 do
        local len = all_data_word[split]:size(1)
        if len % (batch_size) ~= 0 and split < 3 then
           all_data_word[split] = all_data_word[split]:sub(1, batch_size * math.floor(len / batch_size))
           all_label[split] = all_label[split]:sub(1, batch_size * math.floor(len / batch_size))
        end

        x_batches = all_data_word[split]:split(batch_size,1)
        y_batches = all_label[split]:split(batch_size,1)
        nbatches = #x_batches
        self.split_sizes[split] = nbatches
        assert(#x_batches == #y_batches)
        self.all_batches[split] = {x_batches, y_batches}
    end
    self.batch_idx = {0,0,0}
    print(string.format('data load done. Number of batches in train: %d, val: %d, test: %d', self.split_sizes[1], self.split_sizes[2], self.split_sizes[3]))
    collectgarbage()
    return self
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
    local idx = self.batch_idx[split_idx]
    return self.all_batches[split_idx][1][idx], self.all_batches[split_idx][2][idx]
end


function BatchLoaderUnk.text_to_tensor(input_files, out_vocabfile, out_wordfile, out_labelfile, max_sentence_l, max_document_l)
    print('Processing text into tensors...')
    local f, rawdata
    local output_labels = {}
    local output_words = {} -- output word tensors for train/val/test sets
    local vocab_count = {} -- vocab count
    local max_sentence_l_tmp = 0 -- max sentence length of the corpus
    local max_document_l_tmp = 0 -- max document length
    local word_count = 0  --counter for words in a sentence
    local sentence_count = 0
    local document_count = {}
    local idx2word = {}
    local word2idx = {}

    -- first go through train/valid/test to get max sentence/document length
    -- if actual max sentence length is smaller than specified
    -- we use that instead. this is inefficient, but only a one-off thing so should be fine
    -- splitv returns string sequences, split returns table

    for split = 1,3 do
      document_count[split]=0
      for file in lfs.dir(input_files[split]) do
        if #file>15 then
            document_count[split] = document_count[split] + 1
            local f = io.open(input_files[split]..file)
            local content = f:read'*a'
            f:close()
            url, document, summaries = stringx.splitv(content, '\n\n') --others are dropped
            sentence_count = 0
            for _i, line in pairs(stringx.split(summaries, '\n')) do
                word_count = 0
                for _j, word in pairs(stringx.split(line, ' ')) do
                    word_count = word_count+1
                end
                max_sentence_l_tmp = math.max(max_sentence_l_tmp, word_count)
                sentence_count = sentence_count+1
            end
            max_document_l_tmp = math.max(max_document_l_tmp, sentence_count)
        end
      end
    end


    print('After first pass of data, max sentence length is: ' .. max_sentence_l_tmp)
    print('max document length is: ' .. max_document_l_tmp)

    max_sentence_l = math.min(max_sentence_l_tmp, max_sentence_l)
    max_document_l = math.min(max_document_l_tmp, max_document_l)

    for split = 1,3 do
     local doc_id = 0
     output_words[split] = torch.ones(document_count[split], max_document_l, max_sentence_l):long()
     output_labels[split] = torch.ones(document_count[split], max_document_l):long()
     for file in lfs.dir(input_files[split]) do
      if #file>15 then
         doc_id = doc_id+1
         if doc_id % 1000 == 0 then
            collectgarbage()
         end
         local f = io.open(input_files[split]..file)
         local content = f:read'*a'
         f:close()
         url, document, summaries = stringx.splitv(content, '\n\n')
         local sentence_num = 0
         for _i, line in pairs(stringx.split(summaries, '\n')) do
             sentence_num = sentence_num + 1
             if sentence_num > max_document_l then break end
             splitted = stringx.split(line, '*****')
             s, label = splitted[1], splitted[2]

             output_labels[split][doc_id][sentence_num] = tonumber(label)+1
             function append(sentence)
                 local words = {}
                 for _j, word in pairs(stringx.split(line, ' ')) do
                     if word2idx[word]==nil then
                         idx2word[#idx2word + 1] = word
                         word2idx[word] = #idx2word
                     end
                     words[#words + 1] = word2idx[word]
                 end
                 for i = 1, math.min(#words, max_sentence_l) do
                     output_words[split][doc_id][sentence_num][i] = words[i]
                 end
             end
             append(s)
         end
       end
      end
    end
    print "done"
-- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, {idx2word, word2idx})
    print('saving ' .. out_wordfile)
    torch.save(out_wordfile, output_words)
    print('saving ' .. out_labelfile)
    torch.save(out_labelfile, output_labels)
end


return BatchLoaderUnk
