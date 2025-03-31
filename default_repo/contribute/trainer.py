import pytorch_lightning as pl

from default_repo.utils.crossformer.data_tools.data_interface import DataInterface
from default_repo.utils.crossformer.exp import Experiment
from default_repo.utils.crossformer.tools import CustomCallback


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """

    
    # Specify your data exporting logic here

    # Create the data instance
    Data = DataInterface(df=data)
    
    # Create the model instance
    model = Experiment()

    # Load the callbacks, including early stopping and learning rate adjustment
    mycallback = CustomCallback()

    # Create the trainer
    trainer = pl.Trainer(
        accelerator='auto',
        precision=16,
        min_epochs=10,
        max_epochs=10,
        check_val_every_n_epoch=1,
        callbacks=[mycallback],
        fast_dev_run=False,
        # logger=pl.loggers.TensorBoardLogger(
        #     save_dir=result_path,
        #     name='logs',
        #     version=0,
        # ),
    )

    trainer.fit(model, Data)

    trained_model = Experiment.load_from_checkpoint("/home/src/best_model.ckpt")
    test_result = trainer.test(trained_model, Data)
    print(test_result)





