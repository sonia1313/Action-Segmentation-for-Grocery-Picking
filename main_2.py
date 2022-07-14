import yaml
import os
from utils.optoforce_data_loader import load_data
import pytorch_lightning as pl
import importlib.util

from utils.preprocessing import preprocess_dataset

CONFIG_PATH = "config"  # path from root folder


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as f:
        config = yaml.safe_load(f)

        return config


# config = load_config("encoder_decoder_baseline.yaml")
#
# model_pth = config['model']['script_path']
#
# model = getattr(importlib.import_module(model_pth), config['model']['name'])()
# model =importlib.import_module(model_pth)


# odel.loader.exec_module(importlib.util.module_from_spec(model))

def _get_model(module_name, path, pl_class_name):
    model_spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(model_spec)
    # print(module)
    model_spec.loader.exec_module(module)
    return getattr(module, pl_class_name)()


# print(_get_model())

def main():
    config = load_config("encoder_decoder_baseline.yaml")
    #model_pth = config['model']['script_path']

    pl.seed_everything(config['seed'], workers=True)
    trainer = pl.Trainer(default_root_dir=config['train']['checkpoint_path'], gpus=config['train']['gpus'],
                         max_epochs=config['train']['epochs'], deterministic=True)

    model = _get_model(config['model']['module_name'], config['model']['script_path'], config['model']['pl_class_name'])

    X_data, y_data, _ = preprocess_dataset(config['dataset']['preprocess'])
    train_loader, val_loader, test_loader = load_data(X_data, y_data)
    trainer.fit(model, train_loader, val_loader)

    trainer.test(dataloaders=test_loader)

if __name__ == '__main__':
    main()