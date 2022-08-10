import yaml
import os
import pytorch_lightning as pl
import importlib.util
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils.image_data_loader import load_data
from utils.tactile_preprocessing import preprocess_dataset
import wandb as wb
from pytorch_lightning.loggers import WandbLogger
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
    # print(config['seed'])
    # print(type(config['seed']))
    seed = config['seed']
    pl.seed_everything(seed=seed, workers=True)

    n_gpu = 1 if torch.cuda.is_available() else 0

    if n_gpu == 1:
        device = torch.device("cuda:0")
        print(torch.cuda.get_device_properties(device).name)

    model = _get_model(config['model']['module_name'], config['model']['script_path'], config['model']['pl_class_name'])

    model = model(img_size=config['model']['img_size'],
                  input_dim=config['model']['input_dim'],
                  hidden_dim=config['model']['n_hid'],
                  n_layers = config['model']['n_layers'],
                  kernel_size = config['model']['kernel_size'],
                  pool_size = config['model']['pool_size'],
                  lr = config['model']['lr'],
                  exp_name=config['experiment_name'],
                  experiment_tracking=False)

    # load image_data
    file = config['dataset']['preprocess']['image_frames_per_sec']
    dataset = torch.load(file)

    #sanity check
    print(f"no_sequences:{len(dataset)}")
    print(config['dataset']['preprocess']['image_frames_per_sec'])


    wb.init(project =config['project_name'], name=config['experiment_name'], notes=config['notes'],config=config)

    checkpoint_callback = ModelCheckpoint(save_last=True,
                                          monitor="val_loss",
                                          mode="min",
                                          filename=f"{config['experiment_name']}"'-{epoch:02d}-{val_loss:.2f}')
    early_stopping = EarlyStopping(monitor="val_loss", mode="min",patience=5)
    trainer = pl.Trainer(default_root_dir=f"{config['train']['checkpoint_path']}/{config['experiment_name']}",
                         callbacks=[checkpoint_callback, early_stopping],
                         gpus=n_gpu,
                         max_epochs=config['train']['epochs'],
                         deterministic=True,
                         num_sanity_val_steps=0,
                         accelerator="auto",
                         num_nodes=1

                         )

    single = config['dataset']['preprocess']['single']
    clutter = config['dataset']['preprocess']['clutter']
    batch_size = config['train']['batch_size']
    img_size = config['model']['img_size']
    train_loader, val_loader, test_loader = load_data(dataset,
                                                      single=single,
                                                      clutter=clutter,
                                                      batch_size=batch_size,
                                                      size=img_size,
                                                      seed=seed)
    trainer.fit(model, train_loader, val_loader)

    ckpt_path = trainer.checkpoint_callback.last_model_path
    print(ckpt_path)
    trainer.test(dataloaders=test_loader, ckpt_path=ckpt_path)


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
