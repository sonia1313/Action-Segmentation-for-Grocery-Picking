import yaml
import os
import pytorch_lightning as pl
import importlib.util
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.tcn_kfold_optoforce_datamodule  import OpToForceKFoldDataModule, KFoldLoop
from utils.tactile_preprocessing import preprocess_dataset
# import wandb
# from pytorch_lightning.loggers import WandbLogger
import argparse

CONFIG_PATH = "config"  # path from root folder


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as f:
        config = yaml.safe_load(f)

        return config


def _get_model(module_name, path, pl_class_name):
    model_spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(model_spec)
    # print(module)
    model_spec.loader.exec_module(module)
    return getattr(module, pl_class_name)


# print(_get_model())

def main(yaml_file):
    config = load_config(yaml_file)
    # model_pth = config['model']['script_path']
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pl.seed_everything(config['seed'], workers=True)

    n_gpu = 1 if torch.cuda.is_available() else 0

    if n_gpu == 1:
        device = torch.device("cuda:0")
        print(torch.cuda.get_device_properties(device).name)

    model = _get_model(config['model']['module_name'], config['model']['script_path'], config['model']['pl_class_name'])
    # print(model)
    model = model(n_features=config['model']['n_features'],
                  n_hid=config['model']['n_hid'],
                  n_levels=config['model']['n_levels'],
                  kernel_size = config['model']['kernel_size'],
                  dropout=config['model']['dropout'],
                  lr=config['model']['lr'],
                  stride = config['model']['stride'],
                  exp_name=config['experiment_name'])

    file = config['dataset']['preprocess']['tactile_frames_per_sec']
    X_data, y_data = torch.load(file)

    print(f"no_sequences:{len(X_data)}")

    single = config['dataset']['preprocess']['single']
    clutter = config['dataset']['preprocess']['clutter']
    batch_size = config['train']['batch_size']
    seed = config['seed']
    datamodule = OpToForceKFoldDataModule(X_data, y_data,
                                          single=single,
                                          clutter=clutter,
                                          batch_size=batch_size,
                                          seed=seed)

    checkpoint_callback = ModelCheckpoint(save_last=True,
                                          monitor="val_acc",
                                          mode="max",
                                          filename=f"{config['experiment_name']}"'-{epoch:02d}-{val_loss:.2f}')

    trainer = pl.Trainer(default_root_dir=f"{config['train']['checkpoint_path']}/{config['experiment_name']}",
                         callbacks=[checkpoint_callback],
                         gpus=n_gpu,
                         max_epochs=config['train']['epochs'],
                         deterministic=True,
                         num_sanity_val_steps=0,
                         accelerator="auto",
                         num_nodes=1,

                         )

    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = KFoldLoop(num_folds=config['train']['n_kfolds'], export_path=config['train']['kfold_path'],
                                 n_features=config['model']['n_features'],
                                 n_hid=config['model']['n_hid'],
                                 n_levels=config['model']['n_levels'],
                                 kernel_size=config['model']['kernel_size'],
                                 dropout=config['model']['dropout'],
                                 lr=config['model']['lr'],
                                 project_name = config['project_name'],
                                 experiment_name=config['experiment_name'],
                                 config = config,

                                 )
    trainer.fit_loop.connect(internal_fit_loop)

    trainer.fit(model, datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run config file")
    parser.add_argument("-f",
                        "--file",
                        dest="filename",
                        help="experiment configuration file",
                        required=True
                        )
    args = parser.parse_args()

    main(args.filename)
