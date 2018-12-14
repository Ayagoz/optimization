import numpy as np
def stepwise_lr(lr_previous, k, it, decay=0.1, border_for_lr= 5*1e-3):
    if lr_previous < border_for_lr:
        return lr_previous

    if it%k == 0:
        return lr_previous * decay



def exponential_decay(initial_lr, it, k=0.1):
    return initial_lr * np.exp(-k*it)

def cycle(step_size, it,  max_lr=0.9, min_lr=1e-2*5):
    lr_cycle = np.floor(1 + it/float(2*step_size))
    x = np.abs(it/float(step_size) - 2 * lr_cycle + 1)
    return min_lr + (max_lr- min_lr) * np.max([0, 1-x])


def cycle1(step_size, init_lr, it, scale, num_iter=20):
    if it > 2 * step_size:
        x = (it - 2 * step_size)/float(num_iter - 2 * step_size)
        new_lr =init_lr * (1. + (x * (1. - 100.) / 100.)) * scale

    elif it > step_size:
        x = 1. - (it - step_size) / step_size
        new_lr = init_lr * (1. + x * (scale * 100 - 1.)) * scale

    else:
        x = it / step_size
        new_lr = init_lr * (1. + x * (scale * 100 - 1.)) * scale

    if it == num_iter:
        new_lr = 0.
    return new_lr