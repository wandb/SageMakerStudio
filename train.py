# USAGE: python train.py --config configs.py
import gc
import wandb
from absl import app
from absl import flags
import ml_collections
from ml_collections.config_flags import config_flags
from fastai.vision.all import *

from segmentation.camvid_utils import *
from segmentation.train_utils import *
from segmentation.metrics import *


# Config
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
# Flags for sweep
flags.DEFINE_string("backbone", "mobilenetv2_100", "Backbone to be used.")
flags.DEFINE_integer("batch_size", 8, "Batch size to be used.")
flags.DEFINE_integer("image_resize_factor", 4, "Factor for image resizing.")
flags.DEFINE_string("loss_function", False, "Loss function to be used.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning Rate for training the model")
flags.DEFINE_float("weight_decay", 1e-2, "Weight decay rate")


def main(_):
    configs = CONFIG.value
    config.experiment_configs.backbone = FLAGS.backbone
    config.experiment_configs.batch_size = FLAGS.batch_size
    config.experiment_configs.image_resize_factor = FLAGS.image_resize_factor
    config.experiment_configs.loss_function = FLAGS.loss_function
    config.experiment_configs.learning_rate = FLAGS.learning_rate
    config.experiment_configs.weight_decay = FLAGS.weight_decay
    
    wandb_configs = configs.wandb_configs
    experiment_configs = configs.experiment_configs
    loss_alias_mappings = configs.loss_mappings
    inference_config = configs.inference

    run = wandb.init(
        name=wandb_configs.name,
        project=wandb_configs.project,
        entity=wandb_configs.entity,
        job_type=wandb_configs.job_type,
        config=experiment_configs.to_dict(),
    )
    set_seed(wandb.config.seed)

    data_loader, class_labels = get_dataloader(
        artifact_id=configs.wandb_configs.artifact_id,
        batch_size=wandb.config.batch_size,
        image_shape=(wandb.config.image_height, wandb.config.image_width),
        resize_factor=wandb.config.image_resize_factor,
        validation_split=wandb.config.validation_split,
        seed=wandb.config.seed,
    )

    learner = get_learner(
        data_loader,
        backbone=wandb.config.backbone,
        hidden_dim=wandb.config.hidden_dims,
        num_classes=len(class_labels),
        checkpoint_file=None,
        loss_func=loss_alias_mappings[wandb.config.loss_function](axis=1),
        metrics=[DiceMulti(), foreground_acc],
        log_preds=False,
    )

    if wandb.config.fit == "fit":
        learner.fit_one_cycle(
            wandb.config.num_epochs,
            wandb.config.learning_rate,
            wd=wandb.config.weight_decay,
        )
    else:
        learner.fine_tune(
            wandb.config.num_epochs,
            wandb.config.learning_rate,
            wd=wandb.config.weight_decay,
        )

    wandb.log(
        {f"Predictions_Table": table_from_dl(learner, learner.dls.valid, class_labels)}
    )

    # store model checkpoints and JIT
    save_model_to_artifacts(
        learner.model,
        f"Unet_{wandb.config.backbone}",
        image_shape=(wandb.config.image_height, wandb.config.image_width),
        artifact_name=f"{run.name}-saved-model",
        metadata={
            "backbone": wandb.config.backbone,
            "hidden_dims": wandb.config.hidden_dims,
            "input_size": (wandb.config.image_height, wandb.config.image_width),
            "class_labels": class_labels,
        },
    )

    ## Inference benchmark
    model_file = f"Unet_{wandb.config.backbone}_traced.pt"
    learner.model = learner.model.cpu()
    del learner
    gc.collect()
    torch.cuda.empty_cache()
    inference_time = benchmark_inference_time(
        model_file,
        batch_size=inference_config.batch_size,
        image_shape=(wandb.config.image_height, wandb.config.image_width),
        num_warmup_iters=inference_config.warmup,
        num_iter=inference_config.num_iter,
        resize_factor=inference_config.resize_factor,
    )
    wandb.log({"inference_time": inference_time})


if __name__ == "__main__":
    app.run(main)
