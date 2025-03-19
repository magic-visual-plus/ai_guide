from pointcept.models.point_transformer.point_transformer_seg import TransitionDown, Bottleneck
import torch.nn as nn
import torch
from .common import ResMLP
from .lk import se3
import pytorch3d.transforms


class PointTransformerFeat(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        block = Bottleneck
        blocks = [1, 1, 1, 1, 1]
        self.in_channels = in_channels
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(
            block,
            planes[0],
            blocks[0],
            share_planes,
            stride=stride[0],
            nsample=nsample[0],
        )  # N/1
        self.enc2 = self._make_enc(
            block,
            planes[1],
            blocks[1],
            share_planes,
            stride=stride[1],
            nsample=nsample[1],
        )  # N/4
        self.enc3 = self._make_enc(
            block,
            planes[2],
            blocks[2],
            share_planes,
            stride=stride[2],
            nsample=nsample[2],
        )  # N/16
        self.enc4 = self._make_enc(
            block,
            planes[3],
            blocks[3],
            share_planes,
            stride=stride[3],
            nsample=nsample[3],
        )  # N/64
        self.enc5 = self._make_enc(
            block,
            planes[4],
            blocks[4],
            share_planes,
            stride=stride[4],
            nsample=nsample[4],
        )  # N/256
        self.output = nn.Sequential(
            ResMLP(planes[4], 6),
            # nn.ReLU(),
            # nn.Linear(planes[4], planes[4]),
        )          
        pass

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = [
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def forward(self, data_dict):
        p0 = data_dict["coord"]
        x0 = data_dict["feat"]
        o0 = data_dict["offset"].int()
        x0 = p0 if self.in_channels == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x = []
        for i in range(o5.shape[0]):
            if i == 0:
                s_i, e_i, cnt = 0, o5[0], o5[0]
            else:
                s_i, e_i, cnt = o5[i - 1], o5[i], o5[i] - o5[i - 1]
            x_b = x5[s_i:e_i, :].sum(0, True) / cnt
            x.append(x_b)
        x = torch.cat(x, 0)
        x = self.output(x)
        return x
    

class PointMatcher(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.feat_layer = PointTransformerFeat(in_channels)
        self.max_inter = 1
        # self.eta2g = se3.Exp
        
        # self.output = nn.Bilinear(512, 512, 12)
        self.output_ = nn.Linear(512, 12)
        pass

    def output(self, f0, f1):
        return self.output_(f1-f0)

    def eta2g(self, eta):
        mat = eta[:, :9].reshape(-1, 3, 3)
        shift = eta[:, 9:]

        g = torch.eye(4, device=eta.device).unsqueeze(0).repeat(eta.shape[0], 1, 1)
        g[:, :3, :3] = mat
        g[:, :3, 3] = shift
        return g
    
    def eta2g_(self, eta):
        angle = eta[:, :3]
        shift = eta[:, 3:]
        # shift: [B, 3]

        # angle to matrix
        # angle = torch.sigmoid(angle) * torch.pi * 2
        # print(angle)
        # angle = angle.unsqueeze(-1)
        # a = angle[:, 0]
        # b = angle[:, 1]
        # c = angle[:, 2]
        # ca = torch.cos(a)
        # sa = torch.sin(a)
        # cb = torch.cos(b)
        # sb = torch.sin(b)
        # cc = torch.cos(c)
        # sc = torch.sin(c)
        # R = torch.stack(
        #     [
        #         cb * cc, -cb * sc, sb,
        #         ca * sc + sa * sb * cc, ca * cc - sa * sb * sc, -sa * cb,
        #         sa * sc - ca * sb * cc, sa * cc + ca * sb * sc, ca * cb
        #     ], 1
        # ).reshape(-1, 3, 3)
        # R: [B, 3, 3]
        R = pytorch3d.transforms.euler_angles_to_matrix(angle, "XYZ")
        T = shift.unsqueeze(-1)
        # T: [B, 3, 1]
        g = torch.cat([R, T], 2)
        # g: [B, 3, 4]
        g = torch.cat(
            [g, torch.tensor([[[0, 0, 0, 1]]], device=eta.device).repeat(g.shape[0], 1, 1)], 1
        )
        # print(g)
        return g

    def forward(self, x0, feat0, offset0, x1, feat1, offset1):
        g_merge = torch.eye(4, device=x0.device).unsqueeze(0).repeat(offset0.shape[0], 1, 1)
        for i in range(self.max_inter):
            f0 = self.feat_layer(
                {
                    "coord": x0, 
                    "feat": feat0,
                    "offset": offset0
                })
            f1 = self.feat_layer(
                {
                    "coord": x1, 
                    "feat": feat1, 
                    "offset": offset1
                })

            eta = self.output(f0, f1)
            # eta: [B, 6]
            g = self.eta2g(eta)
            # print(torch.svd(g[:, :3, :3])[1])
            # g = eta.reshape(-1, 4, 4)
            # g_merge = torch.matmul(g, g_merge)
            g_merge = torch.matmul(g, g_merge)
            x0s = []
            for i in range(offset0.shape[0]):
                start = offset0[i - 1] if i > 0 else 0
                end = offset0[i]
                x0s.append(se3.transform(g[i], x0[start: end, :].T).T)
                pass
            x0 = torch.cat(x0s, 0)

            # check convergence
            if torch.norm(eta, dim=1).max() < 1e-3:
                break
            pass


        return g_merge

    def compute_loss(self, pred_T, gt_T):
        inv_gt_T = torch.inverse(gt_T)
        A = pred_T.matmul(inv_gt_T)
        I = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)
        A = A[:, :3, :3]
        I = I[:, :3, :3]
        # print(pred_T)
        # print(gt_T)
        loss = torch.nn.functional.mse_loss(A, I, reduction="none")
        loss = loss.sum(dim=(1, 2)).mean()
        loss = torch.nn.functional.mse_loss(A, I, reduction="none")
        loss = loss.sum(dim=(1, 2)).mean()
        return loss