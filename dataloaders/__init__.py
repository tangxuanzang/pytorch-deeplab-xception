from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, suichang_round1
from torch.utils.data import DataLoader, random_split
from mypath import Path


def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset in ['suichang_round1', 'tiny_suichang_round1']:
        base_dir = Path.db_root_dir(args.dataset)
        train_set = suichang_round1.SuichangSegmentation(args, base_dir=base_dir, split='train')
        num_class = train_set.NUM_CLASSES
        if args.do_eval:
            len_train = int(len(train_set) * 0.8)
            len_val = len(train_set) - len_train
            train_set, val_set = random_split(train_set, [len_train, len_val])
        test_set = suichang_round1.SuichangSegmentation(args, base_dir=base_dir, split='test')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs) if args.do_eval else None
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

