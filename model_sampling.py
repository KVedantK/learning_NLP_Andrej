import torch
ckpt = torch.load("bigram.pt")
W, stoi, itos = ckpt["W"], ckpt["stoi"], ckpt["itos"]

g = torch.Generator().manual_seed(42)

for _ in range(10):
    out = []
    ix = 0  # start at "."
    while True:
        logits = W[ix]
        counts = logits.exp()
        probs = counts/counts.sum()
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:   # hit the end token
            break
        out.append(itos[ix])
    print("".join(out))