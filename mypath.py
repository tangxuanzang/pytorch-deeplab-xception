class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/root/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'suichang_round1':
            return '/root/datasets/suichang_round1'
        elif dataset == 'tiny_suichang_round1':
            return '/root/datasets/tiny_suichang_round1'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
