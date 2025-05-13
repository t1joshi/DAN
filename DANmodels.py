import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, WordEmbeddings, read_word_embeddings
from torch.utils.data import Dataset

file_path = 'data/glove.6B.300d-relativized.txt'


import torch
from torch.utils.data import Dataset

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embeddings_class):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        self.word_embeddings_class = word_embeddings_class

        #find max length of a word
        self.max_len = max([len(ex.words) for ex in self.examples])
        self.word_to_index = word_embeddings_class.word_indexer
        self.pad_index = word_embeddings_class.word_indexer.index_of("PAD")
        self.unk_index = word_embeddings_class.word_indexer.index_of("UNK")

        # Pre-processing
        self.preprocessed_data = []
        
        for example in self.examples:
            sentence = example.words
            label = example.label
            
            # Convert words to their corresponding indices upon comparison with Glove embeddings file
            word_indices = [self.word_embeddings_class.word_indexer.index_of(word) if self.word_embeddings_class.word_indexer.index_of(word) != -1 
                            else self.unk_index for word in sentence]
            
            # Apply padding to ensure uniform lengths
            if len(word_indices) < self.max_len:
                word_indices += [self.pad_index] * (self.max_len - len(word_indices))
            else:
                word_indices = word_indices[:self.max_len]
            
            # Convert to tensors
            indices_tensor = torch.tensor(word_indices, dtype=torch.long)
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            
            self.preprocessed_data.append((indices_tensor, label_tensor))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Retrieve the preprocessed data
        indices_tensor, label_tensor = self.preprocessed_data[idx]
        return indices_tensor, label_tensor

# Two-layer neural network 
class NN2DAN(nn.Module):
    def __init__(self, word_embeddings_class,index_to_embedding, hidden_size, dropout_rate=0.3): 
        super(NN2DAN, self).__init__() 
        self.embedding = word_embeddings_class.get_initialized_embedding_layer(frozen = False)
        self.fc1 = nn.Linear(index_to_embedding, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.long() 
        embedded = self.embedding(x)  
        averaged_embedding = embedded.mean(dim=1)  
        x = F.relu(self.fc1(averaged_embedding))  
        x = self.fc2(x)  
        x=self.sigmoid(x)
        
        return x  


# Three-layer neural network 
class NN3DAN(nn.Module):
    def __init__(self,word_embeddings_class, index_to_embedding, hidden_size, dropout_rate=0.3):
        super(NN3DAN, self).__init__()

        self.embedding = word_embeddings_class.get_initialized_embedding_layer( frozen = False)
        self.fc1 = nn.Linear(index_to_embedding, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.long() 
        embedded = self.embedding(x) 
        averaged_embedding = embedded.mean(dim=1) 
        x = F.relu(self.fc1(averaged_embedding))  
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x  
