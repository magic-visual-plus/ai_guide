import torch.nn as nn
import pointcept
import pointcept.models


class PointTransformerPointcept(nn.Module):
    def __init__(self, input_size):
        super(PointTransformerPointcept, self).__init__()
        self.pt = pointcept.models.point_transformer_seg.PointTransformerSeg50(
            in_channels=input_size,
            num_classes=2,
        )
        pass

    def forward(self, x, feat, offset):
        data_dict = {
            "coord": x,
            "feat": feat,
            "offset": offset,
        }
        return self.pt(data_dict)[:, 1]
        pass

class PointTransformerPointceptSmall(nn.Module):
    def __init__(self, input_size):
        super(PointTransformerPointcept, self).__init__()
        self.pt = pointcept.models.point_transformer_seg.PointTransformerSeg26(
            in_channels=input_size,
            num_classes=2,
        )
        pass

    def forward(self, x, feat, offset):
        data_dict = {
            "coord": x,
            "feat": feat,
            "offset": offset,
        }
        return self.pt(data_dict)[:, 1]
        pass
