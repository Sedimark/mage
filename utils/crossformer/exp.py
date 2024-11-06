import pytorch_lightning as pl
import torch
import torch.nn as nn

from default_repo.utils.crossformer.model.crossformer import TimeSeriesTransformer
from default_repo.utils.crossformer.metrics import metric

class Experiment(pl.LightningModule):
    def __init__(self, cfg=None, learning_rate=1e-4, batch=32):
        super(Experiment, self).__init__()

        self.cfg = cfg

        # select the model here: it can be use yaml to load the model config later

        # self.model = TimeSeriesTransformer(
        #                         data_dim=cfg['Data']['dataDim'], 
        #                         in_len=cfg['Experiment']['inLength'], 
        #                         out_len=cfg['Experiment']['outLength'], 
        #                         seg_len=cfg['Model']['segLength'], 
        #                         window_size=cfg['Model']['windowSize'], 
        #                         factor=cfg['Model']['factor'], 
        #                         model_dim=cfg['Model']['modelDim'], 
        #                         feedforward_dim=cfg['Model']['feedforwardDim'], 
        #                         heads_num=cfg['Model']['headNum'], 
        #                         blocks_num=cfg['Model']['layerNum'], 
        #                         dropout=cfg['Model']['dropout'], 
        #                         baseline=cfg['Model']['baseline'],
        #                         learning_rate=cfg['Experiment']['learningRate'],
        #                         batch=cfg['Experiment']['batchSize']
        # )
        self.model = TimeSeriesTransformer(
                                data_dim=5, 
                                in_len=24, 
                                out_len=24, 
                                seg_len=2, 
                                window_size=4, 
                                factor=10, 
                                model_dim=256, 
                                feedforward_dim=512, 
                                heads_num=4, 
                                blocks_num=3, 
                                dropout=0.2, 
                                baseline=False,
                                learning_rate=0.0001,
                                batch=8,
        )

        # Training Parameters
        self.loss = nn.MSELoss()
        self.learning_rate = learning_rate
        self.batch = batch

        self.metricsCache = {
            "mae":[],
            "mse":[],
            "rmse":[],
            "mape":[],
            "mspe":[],
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (x, y, anno) = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y, anno) = batch
        y_hat = self(x)
        [mae,mse,rmse,mape,mspe] = metric(y_hat, y)
        self.metricsCache["mae"].append(mae)
        self.metricsCache["mse"].append(mse)
        self.metricsCache["rmse"].append(rmse)
        self.metricsCache["mape"].append(mape)
        self.metricsCache["mspe"].append(mspe)

    def on_validation_epoch_end(self,):
        [mae,mse,rmse,mape,mspe] = self._cal_avg_metrics()
        self.log('val_mae', mae, prog_bar=True, logger=True)
        self.log('val_mse', mse, prog_bar=True, logger=True)
        self.log('val_rmse', rmse, prog_bar=True, logger=True)
        self.log('val_mape', mape, prog_bar=True, logger=True)
        self.log('val_mspe', mspe, prog_bar=True, logger=True)
        self.metricsCache = {
            "mae":[],
            "mse":[],
            "rmse":[],
            "mape":[],
            "mspe":[],
        }

    def test_step(self, batch, batch_idx):
        (x, y, anno) = batch
        y_hat = self(x)
        [mae,mse,rmse,mape,mspe] = metric(y_hat, y)
        self.metricsCache["mae"].append(mae)
        self.metricsCache["mse"].append(mse)
        self.metricsCache["rmse"].append(rmse)
        self.metricsCache["mape"].append(mape)
        self.metricsCache["mspe"].append(mspe)

    def on_test_epoch_end(self,):
        [mae,mse,rmse,mape,mspe] = self._cal_avg_metrics()
        self.log('test_mae', mae)
        self.log('test_mse', mse)
        self.log('test_rmse', rmse)
        self.log('test_mape', mape)
        self.log('test_mspe', mspe)
        self.metricsCache = {
            "mae":[],
            "mse":[],
            "rmse":[],
            "mape":[],
            "mspe":[],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        return optimizer
    
    def _cal_avg_metrics(self):
        mae = sum(self.metricsCache["mae"]) / len(self.metricsCache["mae"])
        mse = sum(self.metricsCache["mse"]) / len(self.metricsCache["mse"])
        rmse = sum(self.metricsCache["rmse"]) / len(self.metricsCache["rmse"])
        mape = sum(self.metricsCache["mape"]) / len(self.metricsCache["mape"])
        mspe = sum(self.metricsCache["mspe"]) / len(self.metricsCache["mspe"])
        return [mae, mse, rmse, mape, mspe]