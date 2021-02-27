import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:4,:3, :, : ].data.cpu(), 4, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:4], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 4, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:4], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 4, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)
