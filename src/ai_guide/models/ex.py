import pointcept.models
import torch.nn as nn
import torch
from .. import simple_transformer
from .. import cross_transformer
from tqdm import tqdm
import pointcept

def scatter_mean(src, index, dim=-1, output=None):
    if output is None:
        output = torch.zeros_like(src)
    else:
        output = output.zero_()
        pass

    index = index.unsqueeze(-1).expand(-1, -1, src.size(-1))

    count = torch.zeros_like(output)
    output = output.scatter_add_(dim, index, src)
    count = count.scatter_add_(dim, index, torch.ones_like(src))
    output = output / count.clamp(min=1)

    return output


class ResNetBlock(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(ResNetBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
            pass
        
        return x


class PointNetEx(nn.Module):
    def __init__(self, input_size):
        super(PointNetEx, self).__init__()
        self.hidden_size = 256
        self.attention_head_size = 16
        self.embedding_x = nn.Linear(input_size, self.hidden_size)
        self.embedding_x_sampled = nn.Linear(input_size, self.hidden_size)

        self.encoder1 = simple_transformer.MultiLayerTransformer(
            num_layers=6,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            reduction=None,
            use_structure_matrix=False
        )
        self.cross_encoder = cross_transformer.MultiLayerCrossTransformer(
            num_layers=1,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            reduction=None,
            use_structure_matrix=False
        )
        self.encoder2 = simple_transformer.MultiLayerTransformer(
            num_layers=6,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            reduction=None,
            use_structure_matrix=False
        )
        self.cross_decoder = cross_transformer.MultiLayerCrossTransformer(
            num_layers=1,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            reduction=None,
            use_structure_matrix=False
        )
        self.decoder = simple_transformer.MultiLayerTransformer(
            num_layers=6,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            reduction=None,
            use_structure_matrix=False
        )
        self.decoder_out = cross_transformer.MultiLayerCrossTransformer(
            num_layers=1,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            reduction=None,
            use_structure_matrix=False
        )

        self.linear_out = nn.Linear(self.hidden_size, 1)
        self.linear_map = nn.Linear(input_size, self.hidden_size)
        self.map_res = ResNetBlock(self.hidden_size, 8)
        pass

    def forward(self, x, x_sampled, group_index, knn_index, sampled_mask):
        # x: (B, N, D)
        # x_sampled: (B, M, D)
        # group_index: (B, M, G)
        # knn_index: (B, N, K)
        # sampled_mask: (B, M)

        B = x.size(0)
        M = x_sampled.size(1)
        G = group_index.size(2)

        x_ = x
        x = self.embedding_x(x)
        # x: (B, N, H)
        x_sampled = self.embedding_x_sampled(x_sampled)
        # x: (B, M, H)

        group_mask = group_index >= 0
        # group_mask: (B, M, G)
        group_index[torch.logical_not(group_mask)] = 0

        group_index = group_index.reshape(group_index.size(0), -1)
        # group_index: (B, M * G)
        x_group = x.gather(1, group_index.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        # x_group: (B, M * G, H)
        x_group = x_group.reshape(x_group.size(0), group_mask.size(1), -1, x.size(-1))
        # x_group: (B, M, G, H)
        x_group = x_group.masked_fill(torch.logical_not(group_mask.unsqueeze(-1)), 0)
        # x_group: (B, M, G, H)
        x_group = x_group.reshape(-1, x_group.size(-2), x.size(-1))
        # x_group: (B * M, G, H)
        group_mask = group_mask.reshape(-1, group_mask.size(-1))
        # group_mask: (B * M, G)
        x_group = self.encoder1(x_group, attention_mask=group_mask.float())
        # x_group: (B * M, G, H)
        x_sampled = x_sampled.reshape(-1, 1, x_sampled.size(-1))
        # x_sampled: (B * M, 1, H)
        x_sampled = self.cross_encoder(x_sampled, x_group, attention_mask=group_mask.float())
        # x_sampled: (B * M, 1, H)
        x_sampled = x_sampled.reshape(-1, M, x_sampled.size(-1))
        # x_sampled: (B, M, H)
        x_sampled = self.encoder2(x_sampled, attention_mask=sampled_mask)
        # x_sampled: (B, M, H)
        x_sampled = x_sampled.reshape(-1, 1, x_sampled.size(-1))
        # x_sampled: (B * M, 1, H)
        x_group = self.cross_decoder(x_group, x_sampled)
        # x_group: (B * M, G, H)
        x_group = self.decoder(x_group, attention_mask=group_mask.float())
        # x_group: (B * M, G, H)
        x_group = x_group.masked_fill(torch.logical_not(group_mask[..., None]), 0)
        # x_group: (B * M, G, H)
        x_group = x_group.reshape(B, -1, x_group.size(-1))
        # x_group: (B, M * G, H)

        x_middle = torch.zeros_like(x)
        x_middle = scatter_mean(x_group, group_index, dim=1, output=x_middle)
        # x_middle: (B, N, H)
        K = knn_index.size(-1)
        knn_index = knn_index.reshape(B, -1)
        # knn_index: (B , N * K)

        x_knn = torch.gather(x_middle, 1, knn_index.unsqueeze(-1).expand(-1, -1, x_middle.size(-1)))
        # x_knn: (B, N * K, H)
        x_knn = x_knn.reshape(-1, K, x_middle.size(-1))
        # x_knn: (B * N, K, H)
        x = x.reshape(-1, 1, x.size(-1))
        # x: (B * N, 1, H)
        x_output = self.cross_decoder(x, x_knn)
        # x: (B * N, 1, H)
        x_output = x_output.reshape(B, -1, x.size(-1))
        # x: (B, N, H)

        # x_output = self.linear_map(x_)
        # x_output = self.map_res(x_output)
        x_output = self.linear_out(x_output).squeeze(-1)

        return x_output
