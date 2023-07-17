from hyperopt import hp

config = {
    "n_encode_layers": hp.randint("n_encode_layers", 3, 5),
    "lr_model": hp.uniform("lr_model", 1e-6, 0.001),
    "batch_size": hp.choice("batch_size", [128, 256, 512]),
    "n_epochs": hp.randint("epochs", 30, 150),

    'optimizer_class': hp.choice("optimizer_class", ["Adam", "NAdam", "Adamax"]),
    "hyperparameter_tuning": True,
}
