from pytorch_lightning.callbacks import Callback
import torch

def adjust_learning_rate(lradj, epoch, lr):
    if lradj =='type1':
        lr_adjust = {2: lr * 0.5 ** 1, 4: lr * 0.5 ** 2,
                     6: lr * 0.5 ** 3, 8: lr * 0.5 ** 4,
                     10: lr * 0.5 ** 5}
    elif lradj=='type2':
        lr_adjust = {5: lr * 0.5 ** 1, 10: lr * 0.5 ** 2,
                     15: lr * 0.5 ** 3, 20: lr * 0.5 ** 4,
                     25: lr * 0.5 ** 5}
    else:
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        print('Updating learning rate to {}'.format(lr))
    return lr

class CustomCallback(Callback):
    def __init__(self, cfg=None):
        super().__init__()

        self.cfg = cfg

        self.patience = 10 #self.cfg['Experiment']['patience']
        self.best_metrics = {
            "mae": torch.tensor(float('inf')),
            "mse": torch.tensor(float('inf')),
            "rmse": torch.tensor(float('inf')),
            "mape": torch.tensor(float('inf')),
            "mspe": torch.tensor(float('inf')),
        }
        self.counter = 0

    def on_train_start(self, trainer, pl_module):
        print('The model structure:\n {} \n'.format(pl_module.model))
        print('Training is started on the dataset \n')

    def on_train_end(self, trainer, pl_module):
        print('Training is finished! \n')

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.learning_rate = adjust_learning_rate("type3", trainer.current_epoch + 1, pl_module.learning_rate)

    def on_validation_epoch_end(self, trainer, pl_module):
        mae = sum(pl_module.metricsCache["mae"]) / len(pl_module.metricsCache["mae"])
        mse = sum(pl_module.metricsCache["mse"]) / len(pl_module.metricsCache["mse"])
        rmse = sum(pl_module.metricsCache["rmse"]) / len(pl_module.metricsCache["rmse"])
        mape = sum(pl_module.metricsCache["mape"]) / len(pl_module.metricsCache["mape"])
        mspe = sum(pl_module.metricsCache["mspe"]) / len(pl_module.metricsCache["mspe"])

        if mae < self.best_metrics["mae"] and mse < self.best_metrics["mse"]:
            self.best_metrics["mae"] = mae
            self.best_metrics["mse"] = mse
            self.best_metrics["rmse"] = rmse
            self.best_metrics["mape"] = mape
            self.best_metrics["mspe"] = mspe

            self.counter = 0
            # trainer.save_checkpoint(self.cfg['Experiment']['savePath'] + '/best_model.ckpt')
            trainer.save_checkpoint('best_model.ckpt')
            print('\nThe best model is saved at the epoch: {}\n with the bset metrics: \nMAE: {} \nMSE: {}  \nRMSE: {} \nMAPE: {} \nMSPE: {} '.format(trainer.current_epoch, mae, mse, rmse, mape, mspe))
        else:
            self.counter += 1
            if self.counter > self.patience:
                trainer.should_stop = True