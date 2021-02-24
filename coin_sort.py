import torch
import random


# Mostly the example network from pytorch docs
class Net(torch.nn.Module):
    """Demo network."""

    def __init__(self):
        """Initialize the demonstration network with a single output.
        """
        super(Net, self).__init__()
        self.net = torch.nn.ModuleList()
        self.net.append(torch.nn.Linear(in_features = 1, out_features = 1, bias = True))
        self.net.append(torch.nn.Linear(in_features = 1, out_features = 1, bias = True))
        self.net.append(torch.nn.ReLU6())
        self.net.append(torch.nn.Linear(in_features = 1, out_features = 1, bias = True))
        self.net.append(torch.nn.Linear(in_features = 1, out_features = 1, bias = True))

        # Learning can't converge to this.
        # Maybe because there is only a single path through the net?
        with torch.no_grad():
            # TODO FIXME Combine the two pairs of linear layers before and after the ReLU6
            self.net[0].bias[0] = -2 * 0.8038 + 0.8743
            self.net[0].weight[0][0] = 1.0

            self.net[1].bias[0] = 0.0
            self.net[1].weight[0][0] = 2.0 / 0.0705

            self.net[3].bias[0] = -1.0
            self.net[3].weight[0][0] = 1.0

            self.net[4].bias[0] = 0.5
            self.net[4].weight[0][0] = 0.5

    def forward(self, x):
        x = x / 24.26
        for layer in self.net:
            x = layer(x)
        return x


# Radius of the different coins. Since they roll off of the input ramp their edges will be this
# distant from one another.
# Dime, penny, nickel, quarter
coin_widths = [17.9, 19.5, 21.21, 24.26]

net = Net().cuda()
net.train()
#optimizer = torch.optim.Adam(net.parameters())
optimizer = torch.optim.SGD(net.parameters(), lr=10e-7)
loss_fn = torch.nn.L1Loss()

for _ in range(1, 1000):
    batch = torch.tensor(random.choices(coin_widths, k=128)).view(128,1).cuda()
    # Just ask the network to place the coins 
    labels = torch.tensor([coin_widths.index(width) for width in batch]).view(128, 1).cuda()
    out = net.forward(batch)
    loss = loss_fn(out, labels)
    loss.backward()
    optimizer.step()

net.eval()
batch = torch.tensor(random.choices(coin_widths, k=32)).view(32,1).cuda()
# Just ask the network to place the coins 
labels = torch.tensor([coin_widths.index(width) for width in batch]).view(32, 1).cuda()
out = net.forward(batch)
print(f"Final input: {batch}")
print(f"Final output: {out}")
print(f"Final labels: {labels}")
print(f"Final loss is {loss_fn(out, labels)}")
