import torch
from torch import nn


class GrandMasterNet(nn.Module):
    def __init__(self,
                 board_vec_dim: int = 64,
                 moves_vec_dim: int = 5,
                 dropout_p: float = 0.4
                 ):
        super().__init__()
        self.board_net = nn.Sequential(
            nn.Linear(board_vec_dim, board_vec_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(board_vec_dim * 2, board_vec_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout_p), # todo !!!!!!!!!
            nn.Linear(board_vec_dim, board_vec_dim // 2),
            nn.LeakyReLU()

        )

        self.moves_net = nn.Sequential(
            nn.Linear(moves_vec_dim, moves_vec_dim * 4),
            nn.LeakyReLU(),
            nn.Linear(moves_vec_dim * 4, moves_vec_dim * 2),
            nn.LeakyReLU()
        )

        self.output_net = nn.Sequential(
            nn.Linear((board_vec_dim // 2 + moves_vec_dim * 2), (board_vec_dim // 2 + moves_vec_dim * 2) // 2),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout_p), # todo !!!!!!!!!
            nn.Linear((board_vec_dim // 2 + moves_vec_dim * 2) // 2, moves_vec_dim),
            nn.LeakyReLU()
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