
function createVocabulary(fileName)
  -- List Of Words --
  local vocabulary = {}
  local reverseVocabulary = {}
  local index  = 1;
  local maxLength = 0;
  
  for line in io.lines(fileName) do
    -- extract every word --
    local currentLength = 0;
    for word in string.gmatch(line,"[^ ]+") do
      currentLength = currentLength +1; 
      if(vocabulary[word] == nil) then
        vocabulary[word] = index;
        
        reverseVocabulary[index] = word
        index = index + 1;
      end
      
      if currentLength > maxLength then
        maxLength = currentLength;
      end
      
    end
    vocabulary["</S>"] = index
    reverseVocabulary[index] = "</S>"
  end
  
  local vocabularySize = 0;
  
  for _ in pairs(vocabulary) do 
    vocabularySize = vocabularySize + 1 
  end
  
  return vocabulary, reverseVocabulary, maxLength, vocabularySize, vocabulary["</S>"]
end
