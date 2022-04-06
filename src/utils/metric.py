from torch.nn import MSELoss

def loss_fn(y, yhat):
    mse = MSELoss()
    return mse(yhat, y)