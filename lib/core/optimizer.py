import torch.optim as Opt

def get_optimizer(model, args):

    opt_fns = {
        'adam': Opt.Adam(model.parameters(), lr = args.lr_start,weight_decay=1e-4),
        'sgd': Opt.SGD(model.parameters(), lr = args.lr_start,weight_decay=1e-4),
        'adagrad': Opt.Adagrad(model.parameters(), lr = args.lr_start,weight_decay=1e-4)
    }

    return opt_fns.get(args.optimizer, "Invalid Optimizer")
