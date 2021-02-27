import os
import shutil
from collections import OrderedDict
import glob

import torch
import yaml


class Saver(object):
    """
    Save checkpoint and initial parameters configuration.
    """

    def __init__(self, resume, ft):
        _run = 'runs'
        if resume and not ft:
            assert os.path.isfile(resume)
            self.expr_dir = os.path.dirname(resume)
        else:
            self.previous_exprs_dir = sorted(glob.glob(os.path.join(_run, 'expr_*')),
                    key=os.path.getctime)
            run_id = int(self.previous_exprs_dir[-1].split('_')[-1]) + 1 if self.previous_exprs_dir else 0
            self.expr_dir = os.path.join(_run, 'expr_{}'.format(str(run_id)))
        if not os.path.exists(self.expr_dir):
            os.makedirs(self.expr_dir)
        print('The experiment directory is ', self.expr_dir)

    def save_checkpoint(self, state, filename='last.pth', is_best=False, miou=0.0):
        """
        Saves checkpoint to disk
        :param state
        :param filename
        :param is_best
        :param mou
        """
        filename = os.path.join(self.expr_dir, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.expr_dir, 'best.pth'))
            with open(os.path.join(self.expr_dir, 'best_miou.yaml'), 'w') as fp:
                yaml.dump({'miou': miou}, fp)    

    def save_experiment_config(self, args):
        logfile = os.path.join(self.expr_dir, 'parameters.yaml')
        config = OrderedDict()
        config['dataset'] = args.dataset
        config['backbone'] = args.backbone
        config['out_stride'] = args.out_stride
        config['lr'] = args.lr
        config['lr_scheduler'] = args.lr_scheduler
        config['loss_type'] = args.loss_type
        config['epoch'] = args.epochs
        config['base_size'] = args.base_size
        config['crop_size'] = args.crop_size

        with open(logfile, 'w') as fp:
            yaml.dump(config, fp)

    def get_previous_best(self):
        best_miou_path = os.path.join(self.expr_dir, 'best_miou.yaml')
        if os.path.isfile(best_miou_path):
            with open(best_miou_path, 'r') as fp:
                return yaml.safe_load(fp)['miou']
        else:
            return 0.0
        
        
