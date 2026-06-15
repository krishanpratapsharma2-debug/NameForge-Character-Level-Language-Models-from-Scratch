import torch
import torch.nn.functional as F

# ----------------------------
# Load Dataset
# ----------------------------

words = open("names.txt", "r").read().splitlines()

# Character mappings
chars = sorted(list(set("".join(words))))

stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0

itos = {i: ch for ch, i in stoi.items()}

# ----------------------------
# Build Training Set
# ----------------------------

xs = []
ys = []

for word in words:

    chars_seq = ["."] + list(word) + ["."]

    for ch1, ch2 in zip(chars_seq, chars_seq[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)

num_examples = xs.nelement()

# ----------------------------
# Train Neural Bigram Model
# ----------------------------

g = torch.Generator().manual_seed(2147483647)

W = torch.randn(
    (27, 27),
    generator=g,
    requires_grad=True
)

for step in range(100):

    # Forward Pass
    x_encoded = F.one_hot(xs, num_classes=27).float()

    logits = x_encoded @ W
    counts = logits.exp()

    probs = counts / counts.sum(dim=1, keepdim=True)

    loss = (
        -probs[torch.arange(num_examples), ys].log().mean()
        + 0.01 * (W**2).mean()
    )

    # Backward Pass
    W.grad = None
    loss.backward()

    # Gradient Descent
    W.data += -50 * W.grad

print(f"Final Loss: {loss.item():.4f}")

# ----------------------------
# Generate Names
# ----------------------------

g = torch.Generator().manual_seed(2147483647)

for _ in range(10):

    generated_name = []
    current_char = 0

    while True:

        x_encoded = F.one_hot(
            torch.tensor([current_char]),
            num_classes=27
        ).float()

        logits = x_encoded @ W
        counts = logits.exp()

        probs = counts / counts.sum(dim=1, keepdim=True)

        current_char = torch.multinomial(
            probs,
            num_samples=1,
            replacement=True,
            generator=g
        ).item()

        if current_char == 0:
            break

        generated_name.append(itos[current_char])

    print("".join(generated_name))
