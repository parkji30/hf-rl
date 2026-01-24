import torch
from torch import nn
import torch.nn.functional as F

class QNN(nn.Module):
    """
    Neural Net over Q-Table
    """
    def __init__(self):
        
        super().__init__()

        self.state_embedder = nn.Embedding(num_embeddings=64, embedding_dim=4)
        self.ff_layers = nn.Sequential(
            nn.Linear(4, 10),
            nn.GELU(),
            nn.Linear(10, 10),
            nn.GELU(),
            nn.Linear(10, 4) # The action space
        )

    def forward(self, x):
        # Embed our actions 
        embedded_state = self.state_embedder(x)

        # Non-Linear MLP
        ff_layers = self.ff_layers(embedded_state)

        # Softmax to get the probs
        # action_value = self.softmax(ff_layers)
        return ff_layers



    
if __name__ == "__main__":

    index = torch.tensor([5])  # Just the index, shape [1]
    model = QNN(n_layers=5)
    output = model(index)  # Embedding handles the lookup, outputs [1, 4]
    print(output)
    print(f"The best action: {torch.argmax(output)}")