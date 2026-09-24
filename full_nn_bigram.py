'''
Attack plan:

-> Create the vocablury of the words. 
-> Get the integer equivalent of the words
-> add the start and end symbols

'''

import torch
import torch.nn.functional as F

## Creating the database

words = open("names.txt").read().splitlines()
vocab = sorted(list(set("".join(words))))

## String to integer encoder
stoi = {s:i+1 for i,s in enumerate(vocab)}
stoi["."] = 0

## Integer to String decoder
itos = {i:s for s,i in stoi.items()}

## Creating the N matric for bigrams
x,y = [], []
for word in words:
    chs = ["."] + list(word) + ["."]
    for ch1,ch2 in zip(chs, chs[1:]):
        x.append(stoi[ch1])
        y.append(stoi[ch2])

x = torch.tensor(x)
y = torch.tensor(y)
g = torch.Generator().manual_seed(42)
xenc = F.one_hot(x, num_classes=27).float()
W = torch.randn((27,27), requires_grad=True, generator=g)

iterations = 100
num = x.nelement()
for k in range(iterations):
    logits = xenc@W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    loss = -probs[torch.arange(num), y].log().mean()
    print(f"loss for iteration is {loss}")
    W.grad = None
    loss.backward()
    W.data += -70 * W.grad

torch.save({"W": W.detach(), "stoi": stoi, "itos": itos}, "bigram.pt")

        
        