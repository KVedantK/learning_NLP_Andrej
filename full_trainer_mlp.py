import torch
import torch.nn.functional as F


class Network:
    def __init__(self, n_hidden_layers = 100, embedding_dimensions = 25, context_length = 3, iterations = 50000, batch_size=64, lr = 0.1, decay_lr = 0.01):
        self.n_hidden_layers = n_hidden_layers
        self.embedding_dimensions = embedding_dimensions
        self.context_length = context_length
        self.iterations = iterations
        self.batch_size = batch_size
        self.lr = lr
        self.decay_lr = decay_lr

    def build_dataset(self, file_name= "names.txt", n1 = 0.8, n2 = 0.9):
        words = open(file_name, 'r').read().splitlines()
        ## Stois and Itos
        self.chars = sorted(list(set("".join(words))))
        stoi = {s:i+1 for i,s in enumerate(self.chars)}
        stoi["."] = 0
        itos = {i:s for s,i in stoi.items()}

        n1 = int(n1 * len(words))
        n2 = int(n2 * len(words))
        
        def build_tensor_set(words, context_length = self.context_length):
            X, Y = [], []
            for w in words:
                context = context_length * [0]
                for ch in w+".":
                    X.append(context)
                    Y.append(stoi[ch])
                    #print(f"for context {context} ==> {ch}")
                    context = context[1:] + [stoi[ch]]
            
            return torch.tensor(X), torch.tensor(Y)

        Xtrain, Ytrain = build_tensor_set(words[:n1])
        Xval, Yval = build_tensor_set(words[n1:n2])
        Xtest, Ytest = build_tensor_set(words[n2:])

        return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest
    
    def train_network(self, Xtrain, Ytrain):
        C = torch.randn((len(self.chars)+1, self.embedding_dimensions))
        W1 = torch.randn((self.context_length * self.embedding_dimensions, self.n_hidden_layers))
        b1 = torch.randn((self.n_hidden_layers))
        W2 = torch.randn((self.n_hidden_layers, len(self.chars)+1))
        b2 = torch.randn(len(self.chars)+1)

        parameters = [C, W1, b1, W2, b2]

        for p in parameters:
            p.requires_grad = True 
        
        lossi = []
        iterationi = []
        for i in range(self.iterations):
            ix = torch.randint(0, len(Xtrain), (self.batch_size, ))
            embed = C[Xtrain[ix]]
            h = torch.tanh(embed.view(-1, self.context_length * self.embedding_dimensions) @ W1 +b1)
            logits = h @ W2 + b2
            loss = F.cross_entropy(logits, Ytrain[ix])
            lossi.append(loss.item())
            iterationi.append(i)

            for p in parameters:
                p.grad = None

            loss.backward()
            for p in parameters:
                p.data += -self.lr * p.grad if i <30000 else -self.decay_lr * p.grad
        
        return {"embedding_lookup":C, "weights_layer1":W1, "weights_layer2":W2, "bias_l1":b1, "bias_l2":b2}, lossi, iterationi
    
    @torch.no_grad()
    def check_val_loss(self, Xval, Yval, parameters:dict):
        embed = parameters["embedding_lookup"][Xval]
        h = torch.tanh(embed.view(-1, self.context_length * self.embedding_dimensions) @ parameters["weights_layer1"] + parameters["bias_l1"])
        logits = h @ parameters["weights_layer2"] + parameters["bias_l2"]
        loss = F.cross_entropy(logits, Yval)

        return loss
    @torch.no_grad()
    def check_test_loss(self, Xtest, Ytest, parameters:dict):
        embed = parameters["embedding_lookup"][Xtest]
        h = torch.tanh(embed.view(-1, self.context_length * self.embedding_dimensions) @ parameters["weights_layer1"] + parameters["bias_l1"])
        logits = h @ parameters["weights_layer2"] + parameters["bias_l2"]
        loss = F.cross_entropy(logits, Ytest)

        return loss


N = Network()
print(N.build_dataset())
