import yaml
import os
import pytorch_lightning as pl
import importlib.util
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils.conv_lstm_kfold_image_datamodule  import ImageKFoldDataModule, KFoldLoop
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
    model = model(img_size=config['model']['img_size'],
                  input_dim=config['model']['input_dim'],
                  hidden_dim=config['model']['n_hid'],
                  n_layers = config['model']['n_layers'],
                  kernel_size = config['model']['kernel_size'],
                  pool_size = config['model']['pool_size'],
                  lr = config['model']['lr'],
                  exp_name=config['experiment_name'])

    file = config['dataset']['preprocess']['image_frames_per_sec']
    dataset = torch.load(file)

    print(f"no_sequences:{len(dataset)}")

    single = config['dataset']['preprocess']['single']
    clutter = config['dataset']['preprocess']['clutter']
    batch_size = config['train']['batch_size']
    seed = config['seed']
    img_size = config['model']['img_size']
    #print(img_size)
    datamodule = ImageKFoldDataModule(dataset=dataset,
                                      single=single,
                                      clutter=clutter,
                                      batch_size=batch_size,
                                      seed=seed,
                                      img_size=img_size)

    checkpoint_callback = ModelCheckpoint(save_last=True,
                                          monitor="val_loss",
                                          mode="min",
                                          filename=f"{config['experiment_name']}"'-{epoch:02d}-{val_loss:.2f}')
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    trainer = pl.Trainer(default_root_dir=f"{config['train']['checkpoint_path']}/{config['experiment_name']}",
                         callbacks=[checkpoint_callback,early_stopping],
                         gpus=n_gpu,
                         max_epochs=config['train']['epochs'],
                         deterministic=True,
                         num_sanity_val_steps=0,
                         accelerator="auto",
                         num_nodes=1,

                         )

    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = KFoldLoop(num_folds=config['train']['n_kfolds'], export_path=config['train']['kfold_path'],
                                 img_size=config['model']['img_size'],
                                 input_dim=config['model']['input_dim'],
                                 hidden_dim=config['model']['n_hid'],
                                 n_layers=config['model']['n_layers'],
                                 kernel_size=config['model']['kernel_size'],
                                 pool_size=config['model']['pool_size'],
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
