import torch
from torch import nn


class GrandMasterNet(nn.Module):
    def __init__(self,
                 board_vec_dim: int = 64,
                 moves_vec_dim: int = 5
                 ):
        super().__init__()
        self.board_net = nn.Sequential(
            nn.Linear(board_vec_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 16),
            nn.Tanh()
        )

        self.moves_net = nn.Sequential(
            nn.Linear(moves_vec_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
            nn.Tanh()
        )

        self.output_net = nn.Sequential(
            nn.Linear((16 + 4), 32),
            nn.Tanh(),
            nn.Linear(32, 5),
            #nn.Tanh()
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, board_vector, moves_vector):
        x1 = board_vector/12
        x1 = self.board_net(x1)
        x2 = self.moves_net(moves_vector)

        x = torch.cat((x1, x2), dim=1)
        x = self.output_net(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    net = GrandMasterNet()
    b = torch.rand(1, 64)
    m = torch.rand(1, 5)
    out = net(b, m)

    print(out, torch.argmax(out, dim=1))