# USAGE: python train.py --help
import argparse
import wandb
import ml_collections
from fastai.vision.all import *

from segmentation.camvid_utils import *
from segmentation.train_utils import *
from segmentation.metrics import *
from configs import get_config

# load default configs from ml_collection
configs = get_config()

# load independen configs for easy reference
wandb_configs = configs.wandb_configs
experiment_configs = configs.experiment_configs
loss_alias_mappings = configs.loss_mappings
inference_config = configs.inference

def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--image_resize_factor', type=int, default=experiment_configs.image_resize_factor, help='image resize factor')
    argparser.add_argument('--batch_size', type=int, default=experiment_configs.batch_size, help='batch size')
    argparser.add_argument('--seed', type=int, default=experiment_configs.seed, help='random seed')
    argparser.add_argument('--num_epochs', type=int, default=experiment_configs.num_epochs, help='number of training epochs')
    argparser.add_argument('--learning_rate', type=float, default=experiment_configs.learning_rate, help='learning rate')
    argparser.add_argument('--weight_decay', type=float, default=experiment_configs.weight_decay, help='weight decay')
    argparser.add_argument('--backbone', type=str, default=experiment_configs.backbone, help='timm backbone architecture')
    argparser.add_argument('--loss_function', type=str, default=experiment_configs.loss_function, help='Loss function to use')
    return argparser.parse_args()


def main(wandb_configs, experiment_configs, loss_alias_mappings, inference_config):
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
    args = parse_args()
    experiment_configs.update(vars(args))
    main(wandb_configs, experiment_configs, loss_alias_mappings, inference_config)