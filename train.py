import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from moe import MoE

torch.manual_seed(42)

class SampleDataset(Dataset):
    def __init__(self, num_samples=500, input_dim=10, num_classes=5):
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"{device=}")
    input_dim = 32
    hidden_dim = 128
    num_experts = 4
    top_k = 2
    num_classes = 5
    lr = 0.01
    epochs = 50
    batch_size = 1024

    model = MoE(input_dim, hidden_dim, num_experts, top_k, num_classes).to(device); print(model)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = SampleDataset(num_samples=10000, input_dim=input_dim, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_aux_loss = 0.0
        total_usage = torch.zeros(num_experts)

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            logits, aux_loss, dispatch_mask, scores = model(x)
            ce_loss = criterion(logits, y)
            loss = ce_loss + 0.01 * aux_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ce_loss += ce_loss.item()
            epoch_aux_loss += aux_loss.item()
            total_usage += dispatch_mask.sum(dim=0).cpu()

        if epoch % 10 == 0 or epoch < 5:
            num_batches = len(dataloader)
            avg_loss = epoch_loss / num_batches
            avg_ce_loss = epoch_ce_loss / num_batches
            avg_aux_loss = epoch_aux_loss / num_batches

            usage_percent = total_usage / total_usage.sum() * 100
            usage_str = " | ".join([f"E{i}: {int(u.item())} ({p:.1f}%)" for i, (u, p) in enumerate(zip(total_usage, usage_percent))])

            print(f"Epoch {epoch+1:3d} | CE Loss: {avg_ce_loss:.6f} | Aux Loss: {avg_aux_loss:.6f} | Total: {avg_loss:.6f}")
            print(f"Expert usage: {usage_str}")
            print("-" * 70)

if __name__ == "__main__":
    train()