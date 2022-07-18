import yaml
import os
import pytorch_lightning as pl
import importlib.util
import torch
from utils.optoforce_datamodule import OpToForceKFoldDataModule, KFoldLoop
from utils.preprocessing import preprocess_dataset
import argparse

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
                  hidden_size=config['model']['n_hidden_units'],
                  n_layers=config['model']['n_layers'])

    X_data, y_data, _ = preprocess_dataset(config['dataset']['preprocess'])

    datamodule = OpToForceKFoldDataModule(X_data, y_data)

    trainer = pl.Trainer(default_root_dir=config['train']['checkpoint_path'], gpus=n_gpu,
                         max_epochs=config['train']['epochs'],
                         deterministic=True,
                         # limit_train_batches=2,
                         # limit_val_batches=2,
                         # limit_test_batches=2,
                         num_sanity_val_steps=0,
                         accelerator="auto",
                         num_nodes=1
                         )

    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = KFoldLoop(config['train']['n_kfolds'], export_path=config['train']['kfold_path'],
                                 n_features=config['model']['n_features'],
                                 hidden_size=config['model']['n_hidden_units'],
                                 n_layers=config['model']['n_layers']
                                 )
    trainer.fit_loop.connect(internal_fit_loop)

    trainer.fit(model, datamodule)
    #
    # predictions = trainer.predict(model, datamodule)
    # torch.save(predictions, f'inference_results/encoder_decoder_lstm/{config["experiment_name"]}.npy')

    # print(predictions)


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
