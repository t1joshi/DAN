import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, WordEmbeddings, read_word_embeddings
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer

def train_bpe_tokenizer(sentences, vocab_size=30000, min_frequency=2):
    tokenizer = ByteLevelBPETokenizer()
    
    # Train the tokenizer on the input sentences
    tokenizer.train_from_iterator(sentences, vocab_size=vocab_size, min_frequency=min_frequency)
    
    return tokenizer

class SentimentDatasetDANBPE(Dataset):
    def __init__(self, infile, tokenizer, max_len=128):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Pre-processing
        self.preprocessed_data = []
        
        for example in self.examples:
            sentence = example.words
            label = example.label
            
            # Encode the sentence to subword tokens 
            encoded = self.tokenizer.encode(" ".join(sentence))
            subword_indices = encoded.ids  # Get the subword token indices
            
            if subword_indices is None or len(subword_indices) == 0:
                #print(f"Error: Tokenizer returned None or empty subword indices for sentence: {sentence}")
                continue  # Skip this example or handle appropriately

            # Apply padding or truncation to match max_len
            if len(subword_indices) < self.max_len:
                subword_indices += [self.tokenizer.token_to_id('<pad>')] * (self.max_len - len(subword_indices))
            else:
                subword_indices = subword_indices[:self.max_len]
            
            # Convert the list of subword indices and label to tensors
            
            if None in subword_indices:
                #print(f"Error: Found None in subword_indices for sentence: {sentence}")
                subword_indices = [0 if x is None else x for x in subword_indices]  # Replace None with 0
            indices_tensor = torch.tensor(subword_indices, dtype=torch.long)
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            # Store the preprocessed tensors
            

            self.preprocessed_data.append((indices_tensor, label_tensor))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Retrieve the preprocessed feature vector and label for the given index
        indices_tensor, label_tensor = self.preprocessed_data[idx]
        return indices_tensor, label_tensor



# Two-layer fully connected neural network with average embedding input
class NN2DAN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout_rate=0.3): 
        super(NN2DAN, self).__init__()
        # Embedding layer (maps word indices to embeddings)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.long() 
        embedded = self.embedding(x)  
        averaged_embedding = embedded.mean(dim=1)  
        x = F.relu(self.fc1(averaged_embedding))  
        x = self.fc2(x)  
        x=self.sigmoid(x)
        
        return x  # (no softmax here as we are using CrossEntropyLoss)


# Three-layer fully connected neural network with average embedding input
class NN3DAN(nn.Module):
    def __init__(self,vocab_size,embedding_dim, hidden_size, dropout_rate=0.3):
        super(NN3DAN, self).__init__()
        # Embedding layer (maps word indices to embeddings)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim, hidden_size)
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
        
        return x  # Return raw logits (no softmax here as we are using CrossEntropyLoss)
