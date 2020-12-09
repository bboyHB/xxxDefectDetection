from torch import nn
import torch
import numpy as np
from .relation_tool import PositionalEmbedding

class RelationPredictor(nn.Module):
    """
    Relation Module before Standard classification + bounding box regression layers for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(RelationPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.rela_module = RelationModule()
        # self.rela_module2 = RelationModule()

    def forward(self, x, proposals):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        pe = PositionalEmbedding(proposals[0])
        x = self.rela_module((x, pe))
        # x = self.rela_module2((x, pe))
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class RelationModule(nn.Module):
    def __init__(self, n_relations=16, appearance_feature_dim=1024, key_feature_dim=64, geo_feature_dim=64,
                 isDuplication=False):
        super(RelationModule, self).__init__()
        self.isDuplication = isDuplication
        self.Nr = n_relations
        self.dim_g = geo_feature_dim
        self.relation = nn.ModuleList()
        for N in range(self.Nr):
            self.relation.append(RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim))

    def forward(self, input_data):
        if (self.isDuplication):
            f_a, embedding_f_a, position_embedding = input_data
        else:
            f_a, position_embedding = input_data
        isFirst = True
        for N in range(self.Nr):
            if (isFirst):
                if (self.isDuplication):
                    concat = self.relation[N](embedding_f_a, position_embedding)
                else:
                    concat = self.relation[N](f_a, position_embedding)
                isFirst = False
            else:
                if (self.isDuplication):
                    concat = torch.cat((concat, self.relation[N](embedding_f_a, position_embedding)), -1)
                else:
                    concat = torch.cat((concat, self.relation[N](f_a, position_embedding)), -1)
        return concat + f_a


class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=1024, key_feature_dim=64, geo_feature_dim=64):
        super(RelationUnit, self).__init__()
        self.dim_g = geo_feature_dim
        self.dim_k = key_feature_dim
        self.WG = nn.Linear(geo_feature_dim, 1, bias=True)
        self.WK = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WQ = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_a, position_embedding):
        N, _ = f_a.size()

        position_embedding = position_embedding.view(-1, self.dim_g)

        w_g = self.relu(self.WG(position_embedding))
        w_k = self.WK(f_a)
        w_k = w_k.view(N, 1, self.dim_k)

        w_q = self.WQ(f_a)
        w_q = w_q.view(1, N, self.dim_k)

        scaled_dot = torch.sum((w_k * w_q), -1)
        scaled_dot = scaled_dot / np.sqrt(self.dim_k)

        w_g = w_g.view(N, N)
        w_a = scaled_dot.view(N, N)

        w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
        w_mn = torch.nn.Softmax(dim=1)(w_mn)

        w_v = self.WV(f_a)

        w_mn = w_mn.view(N, N, 1)
        w_v = w_v.view(N, 1, -1)

        output = w_mn * w_v

        output = torch.sum(output, -2)
        return output