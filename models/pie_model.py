import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from pytorch_msssim import ms_ssim

class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super(AttentionBlock, self).__init__()
        self._spatial_attention_conv = nn.Conv2d(2, dim, kernel_size=3, padding=1)

        # Channel attention MLP
        self._channel_attention_conv0 = nn.Conv2d(1, dim, kernel_size=1, padding=0)
        self._channel_attention_conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)

        self._out_conv = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor):
        if len(x.shape) != 4:
            raise ValueError(f"Expected [B, C, H, W] input, got {x.shape}.")

        # Spatial attention
        mean = torch.mean(x, dim=1, keepdim=True)  # Mean/Max on C axis
        max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = torch.cat([mean, max], dim=1)  # [B, 2, H, W]
        spatial_attention = self._spatial_attention_conv(spatial_attention)
        spatial_attention = torch.sigmoid(spatial_attention) * x

        # Channel attention. 
        channel_attention = torch.relu(self._channel_attention_conv0(mean))
        channel_attention = self._channel_attention_conv1(channel_attention)
        channel_attention = torch.sigmoid(channel_attention) * x

        attention = torch.cat([spatial_attention, channel_attention], dim=1)  # [B, 2*dim, H, W]
        attention = self._out_conv(attention)
        return x + attention


# TODO: This is not named in the paper right?
# It is sort of the InverseResidualBlock but w/o the Channel and Spatial Attentions and without another Conv after ReLU
class InverseBlock(nn.Module):
    def __init__(self, input_channels: int, channels: int):
        super(InverseBlock, self).__init__()

        self._conv0 = nn.Conv2d(input_channels, channels, kernel_size=1)
        self._dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self._conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self._conv2 = nn.Conv2d(input_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        features = self._conv0(x)
        features = F.elu(self._dw_conv(features))  # TODO: Paper is ReLU, authors do ELU
        features = self._conv1(features)

        # TODO: The BaseBlock has residuals and one path of convolutions, not 2 separate paths - is this different on purpose?
        x = torch.relu(self._conv2(x))
        return x + features


class BaseBlock(nn.Module):
    def __init__(self, channels: int):
        super(BaseBlock, self).__init__()

        self._conv0 = nn.Conv2d(channels, channels, kernel_size=1)
        self._dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self._conv1 = nn.Conv2d(channels, channels, kernel_size=1)

        self._conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self._conv3 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        features = self._conv0(x)
        features = F.elu(self._dw_conv(features))  # TODO: ELU or ReLU?
        features = self._conv1(features)
        x = x + features

        features = F.elu(self._conv2(x))
        features = self._conv3(features)
        return x + features


class AttentionTail(nn.Module):
    def __init__(self, channels: int):
        super(AttentionTail, self).__init__()

        self._conv0 = nn.Conv2d(channels, channels, kernel_size=7, padding=3)
        self._conv1 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self._conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        attention = torch.relu(self._conv0(x))
        attention = torch.relu(self._conv1(attention))
        attention = torch.sigmoid(self._conv2(attention))
        return x * attention


class PIENet(pl.LightningModule):
    def __init__(self, input_channels: int, output_channels: int, encoder_dims: List[int], decoder_dims: List[int]):
        super(PIENet, self).__init__()
        if len(encoder_dims) != len(decoder_dims) + 1 or len(decoder_dims) < 1:
            raise ValueError(f"Unexpected encoder and decoder dims: {encoder_dims}, {decoder_dims}.")
        if input_channels != output_channels:
            raise NotImplementedError()
        # TODO: We will need an explicit decoder head, consider Unshuffle & Shuffle
        encoders = []
        for i, encoder_dim in enumerate(encoder_dims):
            input_dim = input_channels if i == 0 else encoder_dims[i - 1]
            encoders.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, encoder_dim, kernel_size=3, padding=1),
                    BaseBlock(encoder_dim),
                    BaseBlock(encoder_dim),
                    AttentionBlock(encoder_dim),
                )
            )
        self._encoders = nn.ModuleList(encoders)
        decoders = []
        for i, decoder_dim in enumerate(decoder_dims):
            input_dim = encoder_dims[-1] if i == 0 else decoder_dims[i - 1] + encoder_dims[-i - 1]
            decoders.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, decoder_dim, kernel_size=3, padding=1),
                    BaseBlock(decoder_dim),
                    BaseBlock(decoder_dim),
                    AttentionBlock(decoder_dim),
                )
            )
        self._decoders = nn.ModuleList(decoders)
        self._inverse_bock = InverseBlock(encoder_dims[0] + decoder_dims[-1], output_channels)
        self._attention_tail = AttentionTail(output_channels)

    def forward(self, x: torch.Tensor):
        global_residual = x
        encoder_outputs = []
        for i, encoder in enumerate(self._encoders):
            x = encoder(x)
            if i != len(self._encoders) - 1:
                encoder_outputs.append(x)
                x = F.max_pool2d(x, kernel_size=2)
        for i, decoder in enumerate(self._decoders):
            x = decoder(x)
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x = torch.cat([x, encoder_outputs.pop()], dim=1)
        x = self._inverse_bock(x)
        x = self._attention_tail(x)
        return x + global_residual
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        msssim_loss = 1 - ms_ssim(y_hat, y, data_range=1, size_average=True)
        l1_loss = F.l1_loss(y_hat, y)
        loss = msssim_loss + l1_loss
        self.log('train_loss', loss, logger=True, prog_bar=True)
        self.log('train_msssim_loss', msssim_loss, logger=True, prog_bar=True)
        self.log('train_l1_loss', l1_loss, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        msssim_loss = 1 - ms_ssim(y_hat, y, data_range=1, size_average=True)
        l1_loss = F.l1_loss(y_hat, y)
        loss = msssim_loss + l1_loss
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_msssim_loss', msssim_loss, prog_bar=True)
        self.log('val_l1_loss', l1_loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }


if __name__ == "__main__":
    import torch
    from thop import profile

    # model = PIENet(1, 1, [32, 64, 128], [64, 32])
    model = PIENet(1, 1, [16, 32, 64], [32, 16])
    inputs = torch.randn(1, 1, 256, 256)
    flops, params = profile(model, inputs=(inputs,))
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")