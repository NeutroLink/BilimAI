"""LinkResNet — the ACTUAL architecture behind our production detector. Vendored 2026-08-22.

Source: https://github.com/ai-forever/SEGM-model (segm/models.py), MIT, by ai-forever. Unmodified except
for the `return_logits` flag added to __init__/forward (see below) and this header.

WHY VENDORED: `models/readingpipeline/segm/segm_model.onnx` is our production line detector, and
`ai-forever/ReadingPipeline-notebooks` also publishes `segm/segm_model.ckpt` — the PyTorch weights.
Verified 2026-08-22: LinkResNet(3, 3, encoder="resnet50") loads that checkpoint with strict=True,
0 missing / 0 unexpected, and its output matches the production ONNX to max |delta| 1.5e-5 with
100.000 % agreement on the thresholded word mask. It IS the shipped model, and it is trainable.

`eval/detectors/train_segm.py:5` claims "the RP checkpoint exists only as ONNX, so ... we train from
scratch". That is false, and it cost six segmenter runs (segm_ft_v1..v6, best exam F1 0.694 vs RP's
0.892) which additionally used a SMALLER network (smp.Linknet/resnet34, 21.8 M) than the one they were
trying to beat (LinkResNet/resnet50, 28.8 M).

RETURN_LOGITS: the published forward ends in nn.Sigmoid(), so it emits probabilities. Training with
BCEWithLogitsLoss on that output would apply the sigmoid TWICE and silently learn nothing useful, and
BCELoss on probabilities is unsafe under fp16 autocast. So `return_logits=True` skips the final
sigmoid for training; export keeps it ON so the ONNX stays a drop-in for bilimai/detector.py, which
thresholds raw probabilities at 0.8.
"""
"""
all credits to @nizhib
"""
import torch.nn as nn
from torchvision.models.resnet import \
    resnet18,\
    resnet34,\
    resnet50,\
    resnet101,\
    resnet152
from torch.nn import Conv2d

nonlinearity = nn.ReLU


ENCODERS = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152
}


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class LinkResNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, dropout2d_p=0.5,
                 pretrained=True, encoder='resnet50', return_logits=False):
        assert input_channels > 0
        assert encoder in ENCODERS
        super().__init__()

        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]

        resnet = ENCODERS[encoder](pretrained=pretrained)

        if input_channels != 3:
            resnet.conv1 = Conv2d(input_channels, 64, kernel_size=(7, 7),
                                  stride=(2, 2), padding=(3, 3), bias=False)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dropout2d1 = nn.Dropout2d(p=dropout2d_p)
        self.dropout2d2 = nn.Dropout2d(p=dropout2d_p)
        self.dropout2d3 = nn.Dropout2d(p=dropout2d_p)

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, output_channels, 2, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.return_logits = return_logits   # True = skip the final sigmoid (training); see header

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + self.dropout2d1(e3)
        # d4 = e3
        d3 = self.decoder3(d4) + self.dropout2d2(e2)
        d2 = self.decoder2(d3) + self.dropout2d3(e1)
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5 if self.return_logits else self.sigmoid(f5)
