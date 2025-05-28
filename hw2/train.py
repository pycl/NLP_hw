import torch
import torch.optim as optim
from model import BigramLanguageModel, block_size 

# --- Training Hyperparameters ---
batch_size = 16  
max_iters = 5000  
eval_interval = 500  
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200  


torch.manual_seed(1337)


text = open('ozon_reviews.txt', 'r', encoding='utf-8').read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.8 * len(data))  # 80% for train, rest for val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    #Generate a small batch of data of inputs x and targets y.
    current_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(current_data) - block_size, (batch_size,))
    x = torch.stack([current_data[i:i + block_size] for i in ix])
    y = torch.stack([current_data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model_to_eval):
    out = {}
    model_to_eval.eval() 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model_to_eval(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model_to_eval.train()
    return out

if __name__ == '__main__':
    model = BigramLanguageModel(vocab_size, device) # Pass vocab_size and device
    m = model.to(device)
    print(f"{sum(p.numel() for p in m.parameters()) / 1e6:.6f} M parameters")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for iter_num in range(max_iters):
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            estimated_losses = estimate_loss(m)
            print(f"step {iter_num}: train loss {estimated_losses['train']:.4f}, "
                  f"val loss {estimated_losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\nTraining complete.\n")
    # Generate text from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(m.generate(context, max_new_tokens=2000)[0].tolist())
    print("Generated text:")
    print(generated_text) 