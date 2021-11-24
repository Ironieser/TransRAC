"""
RepNetPeriodEstimator

Credit:
https://openaccess.thecvf.com/content_CVPR_2020/papers/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.pdf

Note:
    conv feature extractor is different from that used in original paper
    check dilation on temporal 3d conv
    pairwise_l2_distance transpose for b
    input projection kernel_regularizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, wide_resnet50_2
from torch.cuda.amp import autocast, GradScaler
from typing import Callable
from torchvision.transforms import ToPILImage


class RepNetPeriodEstimator(nn.Module):
    """
    RepNetPeriodEstimator
    """

    def __init__(self,
                 num_frames: int = 64,
                 image_size: int = 112,
                 temperature: float = 13.544,
                 dropout_rate: float = 0.25,
                 temporal_conv_channels: int = 512,
                 temporal_conv_kernel_size: int = 3,
                 temporal_conv_dilation_rate: int = 3,
                 conv_channels: int = 32,
                 conv_kernel_size: int = 3,
                 transformer_layers_config: tuple = ((512, 4, 512),),
                 transformer_dropout_rate: float = 0.0,
                 transformer_reorder_ln: bool = True,
                 period_fc_channels: tuple = (512, 512),
                 within_period_fc_channels: tuple = (512, 512)
                 ):
        super().__init__()

        # model parameters
        self.num_frames = num_frames
        self.image_size = image_size

        self.temperature = temperature

        self.dropout_rate = dropout_rate

        self.temporal_conv_channels = temporal_conv_channels
        self.temporal_conv_kernel_size = temporal_conv_kernel_size
        self.temporal_conv_dilation_rate = temporal_conv_dilation_rate

        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        # Transformer config in form of (channels, heads, bottleneck channels).
        self.transformer_layers_config = transformer_layers_config
        self.transformer_dropout_rate = transformer_dropout_rate
        self.transformer_reorder_ln = transformer_reorder_ln

        self.period_fc_channels = period_fc_channels
        self.within_period_fc_channels = within_period_fc_channels

        # get resnet50 backbone, drop layers down to Conv3 Bottleneck 2
        self.base_model = get_base_model(wide=False)

        # this is a fix, dilation doesn't work like it does in tf
        self.temporal_conv_dilation_rate = 1

        # temporal conv layers
        self.temporal_conv_layers = nn.ModuleList()
        self.temporal_conv_layers.append(nn.Conv3d(in_channels=1024,
                                                   out_channels=self.temporal_conv_channels,
                                                   kernel_size=self.temporal_conv_kernel_size,
                                                   padding=1,
                                                   dilation=(self.temporal_conv_dilation_rate, 1, 1)
                                                   ))

        self.temporal_bn_layers = nn.ModuleList()
        tmplist = [nn.BatchNorm3d(num_features=512) for _ in self.temporal_conv_layers]
        self.temporal_bn_layers.append(tmplist[0])

        self.conv_3x3_layer = nn.Conv2d(in_channels=1,
                                        out_channels=self.conv_channels,
                                        kernel_size=self.conv_kernel_size,
                                        padding=1)

        channels = self.transformer_layers_config[0][0]
        # how many in features
        self.input_projection = nn.Linear(in_features=2048,
                                          out_features=channels,
                                          bias=True
                                          )

        self.input_projection2 = nn.Linear(in_features=2048,
                                           out_features=channels,
                                           bias=True
                                           )
        self.input_projection3 = nn.Linear(in_features=2048,
                                           out_features=channels,
                                           bias=True
                                           )
        length = self.num_frames
        self.pos_encoding = torch.empty(1, length, 1).normal_(mean=0, std=0.02)
        self.pos_encoding.requires_grad = True

        self.pos_encoding2 = torch.empty(1, length, 1).normal_(mean=0, std=0.02)
        self.pos_encoding2.requires_grad = True

        self.pos_encoding3 = torch.empty(1, length, 1).normal_(mean=0, std=0.02)
        self.pos_encoding3.requires_grad = True

        self.transformer_layers = nn.ModuleList()
        for d_model, num_heads, dff in self.transformer_layers_config:
            tfel = nn.TransformerEncoderLayer(d_model=d_model,
                                              nhead=num_heads,
                                              dim_feedforward=dff,
                                              dropout=self.transformer_dropout_rate)
            self.transformer_layers.append(tfel)

        # self.transformer_layers2 = nn.ModuleList()
        # for d_model, num_heads, dff in self.transformer_layers_config:
        #     tfel = nn.TransformerEncoderLayer(d_model=d_model,
        #                                       nhead=num_heads,
        #                                       dim_feedforward=dff,
        #                                       dropout=self.transformer_dropout_rate)
        #     self.transformer_layers.append(tfel)

        self.dropout_layer = nn.Dropout(self.dropout_rate)

        # density prediction module
        num_preds = 1
        self.fc_layers = nn.ModuleList()

        for channels in self.period_fc_channels:
            self.fc_layers.append(
                nn.Linear(in_features=channels,
                          out_features=channels)

            )
            self.fc_layers.append(nn.ReLU())

        self.fc_layers.append(
            nn.Linear(in_features=self.period_fc_channels[0],
                      out_features=num_preds)
        )
        # density2
        self.transformer_layers2 = self.transformer_layers
        self.fc_layers2 = self.fc_layers

        # density3
        self.transformer_layers3 = self.transformer_layers
        self.fc_layers3 = self.fc_layers

    def forward(self, x: torch.Tensor, epoch=0) -> tuple:
        """

        Args:
            x: input images x: [B, F, C, H, W] -> [B, F, 3, 224, 224]
        Returns:
            x: tensor shape:[b,f,32]
            within_period_x:tensor shape:[b,f]
            final_embs: tensor shape:[b,f,512]
            sim_matrix: tensor shape:[b,1,f,f]
        """
        # Ensure usage of correct batch size
        x = self.preprocess(x)  # => [B, F, 3, 112 112]
        b = x.shape[0]
        x = torch.reshape(x, [-1, 3, self.image_size, self.image_size])  # => [b*f,3,h,w]
        # Conv feature extractor
        # if epoch <50:
        # with torch.no_grad():
        #     x = self.base_model(x)  # =>[b*f,1024,h,w]
        # else:
        #     with autocast():
        #         x = self.base_model(x)  # =>[b*f,1024,h,w]
        with autocast():
            x = self.base_model(x)  # =>[b*f,1024,h,w]
            x = torch.reshape(x, [b, -1, 1024, 7, 7])  # => [batch_size,  f, 1024 ,h, w]
            c = x.shape[2]
            h = x.shape[3]
            w = x.shape[4]
            x = x.transpose(1, 2)  # => [batch_size, 1024 , f, h, w]
            # x = torch.Size([20, 1024, 64, 7, 7])

            for bn_layer, conv_layer in zip(self.temporal_bn_layers,
                                            self.temporal_conv_layers):
                x = conv_layer(x)
                x = bn_layer(x)
                F.relu(x)

                # x = torch.Size([20, 512, 64, 7, 7])
                x, _ = torch.max(x, dim=3)
                x, _ = torch.max(x, dim=3)
                # x = torch.Size([20, 512, 64])
                final_embs = x.permute(0, 2, 1)

                # get self smimillarity matrix
                x = get_sims(x, self.temperature)
                sim_matrix = x
                # x = torch.Size[20, 64, 64, 1]
                x = x.permute([0, 3, 1, 2])
                # x = torch.Size[20, 1, 64, 64]

                x = F.relu(self.conv_3x3_layer(x))  # => [b,c,f,f]
                # x = torch.Size[20, 32, 64, 64]
                x = x.transpose(1, 2)  # => [b,f,c,f]
                x = torch.reshape(x, [b, self.num_frames, -1])
                # x = torch.Size[20, 64, 2048]
                x2 = x
                x3 = x
                # density1 prediction
                x = self.input_projection(x)
                x += self.pos_encoding.to(x.device)
                # x = torch.Size[20, 64, 512]
                for transformer_layer in self.transformer_layers:
                    x = transformer_layer(x)
                # x = torch.Size[20, 64, 512]
                # x = torch.Size[20, 64, 512]
                x = torch.reshape(x, [b, self.num_frames, -1])

                for fc_layer in self.fc_layers:
                    x = self.dropout_layer(x)
                    x = fc_layer(x)

                # density2 prediction
                x2 = self.input_projection2(x2)
                x2 += self.pos_encoding2.to(x2.device)

                for transformer_layer in self.transformer_layers2:
                    x2 = transformer_layer(x2)

                x2 = torch.reshape(x2, [b, self.num_frames, -1])

                for fc_layer in self.fc_layers2:
                    x2 = self.dropout_layer(x2)
                    x2 = fc_layer(x2)

                # density3 prediction
                x3 = self.input_projection3(x3)
                x3 += self.pos_encoding3.to(x3.device)

                for transformer_layer in self.transformer_layers3:
                    x3 = transformer_layer(x3)

                x3 = torch.reshape(x3, [b, self.num_frames, -1])

                for fc_layer in self.fc_layers3:
                    x3 = self.dropout_layer(x3)
                    x3 = fc_layer(x3)

                return x.reshape([b, -1]), x2.reshape([b, -1]), x3.reshape([b, -1]), final_embs, sim_matrix.permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    2)
            # return x, within_period_x

    def preprocess(self, imgs: torch.Tensor):
        """
        Preprocess input images
        :param imgs: images to preprocess
        :return: preprocessed images
        """
        imgs = F.interpolate(imgs, size=[3, self.image_size, self.image_size])
        return imgs


def get_base_model(wide: bool = True):
    """
    Get backbone for RepNetEstimator
    :param wide: whether to use wide resent 50 or nor
    :return: Resnet base model for backbone
    """

    if wide:
        base_model = wide_resnet50_2(pretrained=True)
    else:
        base_model = resnet50(pretrained=True)
    base_model.fc = nn.Identity()
    base_model.avgpool = nn.Identity()
    base_model.layer4 = nn.Identity()
    base_model.layer3[3] = nn.Identity()
    base_model.layer3[4] = nn.Identity()
    base_model.layer3[5] = nn.Identity()
    return base_model


def pairwise_l2_distance(a: torch.Tensor, b: torch.Tensor):
    """
    Computes pairwise distances between all rows of a and all rows of b.
    :param a: tensor
    :param b: tensor
    :return pairwise distance
    """
    norm_a = torch.sum(torch.square(a), dim=0)
    norm_a = torch.reshape(norm_a, [-1, 1])
    norm_b = torch.sum(torch.square(b), dim=0)
    norm_b = torch.reshape(norm_b, [1, -1])
    a = torch.transpose(a, 0, 1)
    zero_tensor = torch.zeros(64, 64).to(norm_a.device)
    dist = torch.maximum(norm_a - 2.0 * torch.matmul(a, b) + norm_b, zero_tensor)
    return dist


def get_sims(embs: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Calculates self-similarity between batch of sequence of embeddings
    :param embs: embeddings
    :param temperature: temperature
    :return self similarity tensor
    """

    batch_size = embs.shape[0]
    seq_len = embs.shape[2]
    embs = torch.reshape(embs, [batch_size, -1, seq_len])

    def _get_sims(embs: torch.Tensor):
        """
        Calculates self-similarity between sequence of embeddings
        :param embs: embeddings
        """

        dist = pairwise_l2_distance(embs, embs)
        sims = -1.0 * dist
        return sims

    sims = map_fn(_get_sims, embs)
    # sims = torch.Size[20, 64, 64]
    sims /= temperature
    sims = F.softmax(sims, dim=-1)
    sims = sims.unsqueeze(dim=-1)
    return sims


def map_fn(fn: Callable, elems: torch.Tensor) -> torch.Tensor:
    """
    Transforms elems by applying fn to each element unstacked on dim 0.
    :param fn: function to apply
    :param elems: tensor to transform
    :return: transformed tensor
    """

    sims_list = []
    for i in range(elems.shape[0]):
        sims_list.append(fn(elems[i]))
    sims = torch.stack(sims_list)
    return sims


def test():
    """
    Test for RepNetEstimator Model
    :return: nothing
    """
    device = 'cuda'
    model = RepNetPeriodEstimator()
    model = torch.nn.DataParallel(model.to(device), device_ids=[0])
    x = torch.randn(2, 64, 3, 224, 224).cuda()
    out = model(x)

    # sim = out[3]
    show = ToPILImage()
    pic1 = out[3].permute(0, 3, 1, 2)[0]
    pic2 = out[3].permute(0, 3, 1, 2)[1]
    show(pic1).show()
    show(pic2).show()
    expected_shapes = [[2, 64, 32],
                       [2, 64, 1],
                       [2, 64, 512]]
    out_names = ["x", "within_period_x", "final_embs"]

    for os, es, ons in zip(out, expected_shapes, out_names):
        for i in range(3):
            assert os.shape[0] == es[0], "Mismatch in shape for output {} at dim {}, expected {}, got {}".format(
                ons[i],
                i,
                es[0],
                os.shape[0])

    print("Got all expected shapes, test passed")

# if __name__ == "__main__":
# #     test()
#
