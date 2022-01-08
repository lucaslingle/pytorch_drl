import importlib


def get_optimizer(model, cls_name, cls_args):
    module = importlib.import_module('torch.optim')
    cls = getattr(module, cls_name)
    if cls_name == 'AdamW':
        wd = cls_args.pop('wd')
        param_groups = get_weight_decay_param_groups(model, wd)
        return cls(param_groups, **cls_args)
    return cls(model.parameters(), **cls_args)


def get_scheduler(optimizer, cls_name, cls_args):
    if cls_name == 'None':
        return None
    module = importlib.import_module('torch.optim.lr_scheduler')
    cls = getattr(module, cls_name)
    return cls(optimizer, **cls_args)


def get_weight_decay_param_groups(model, wd):
    apply_decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif len(param.shape) == 1:
            no_decay.append(param)
        else:
            apply_decay.append(param)
    return [
        {'params': apply_decay, 'weight_decay': wd},
        {'params': no_decay, 'weight_decay': 0.0}
    ]
