from modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, in_channels, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(in_channels, output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(in_channels, output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
