import numpy as np


def ring(z):
    assert z.shape[0] == 1
    e = np.stack([
        np.cos(z[0, :]),
        np.sin(2 * z[0, :]),
        np.sin(z[0, :]),
    ], axis=0)
    return e

def swiss_roll(z):
    assert z.shape[0] == 2
    e = np.stack([
        z[0, :] * np.cos(z[0, :]) / 2,
        z[1, :],
        z[0, :] * np.sin(z[0, :]) / 2,
    ], axis=0)
    return e

def identity(z):
    return z