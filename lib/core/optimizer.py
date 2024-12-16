import torch.optim as Opt

def get_optimizer(model, solver):

    opt_fns = {
        'adam': Opt.Adam(model.parameters(), lr = solver.LR_START,weight_decay=solver.WEIGHT_DECAY),
        'sgd': Opt.SGD(model.parameters(), lr = solver.LR_START,weight_decay=solver.WEIGHT_DECAY),
        'adagrad': Opt.Adagrad(model.parameters(), lr = solver.LR_START,weight_decay=solver.WEIGHT_DECAY)
    }

    return opt_fns.get(solver.NAME, "Invalid Optimizer")
