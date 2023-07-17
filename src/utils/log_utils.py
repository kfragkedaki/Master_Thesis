import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import torch

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.functions import get_inner_model


def plot_encoder_data(figures, tb_logger, step=0, title="Heatmap Figures"):
    fig, axs = plt.subplots(1, len(figures), layout="constrained")

    # Create the heatmap
    img = None
    for index, (name, figure) in enumerate(figures.items()):
        img = axs[index].imshow(figure, cmap="coolwarm", interpolation="nearest")
        axs[index].set_title(name + " " + str(step))

        # Add text annotations to show the heatmap values
        for i in range(figure.shape[0]):
            for j in range(figure.shape[1]):
                text = axs[index].text(
                    j,
                    i,
                    f"{figure[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=6,
                )

    fig.colorbar(img, shrink=0.6)

    # Convert the figure to a numpy array
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert the image to a torch tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()

    # Write the image to TensorBoard
    tb_logger["writer"].add_image(title, image_tensor, global_step=step)

    plt.close(fig)


def plot_attention_weights(model, tb_logger, step=0):
    for layer_idx in range(model.n_encode_layers):
        attention_layer = model.encoder.layers[layer_idx][0].module
        # First instance of the batch
        att = attention_layer.get_attention_weights()[:, 0, :, :].detach().cpu().numpy()

        # Plot the attention weights
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Layer {layer_idx + 1} Attention Weights", fontsize=16)

        for head in range(attention_layer.num_heads):
            ax = axs[head // 4, head % 4]
            heatmap = ax.imshow(att[head], cmap="coolwarm", interpolation="nearest")

            # Set axis labels and title
            ax.set_xlabel("To Node", fontsize=12)
            ax.set_ylabel("From Node", fontsize=12)
            ax.set_title(f"Attention Head {head + 1})", fontsize=12)

            # Add text annotations to show the attention weight values
            for i in range(att[head].shape[0]):
                for j in range(att[head].shape[1]):
                    text = ax.text(
                        j,
                        i,
                        f"{att[head][i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        # Add colorbar for reference
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position of the colorbar
        cbar = plt.colorbar(heatmap, cax=cbar_ax)
        cbar.set_label("Attention Weight", fontsize=12)

        # Convert the figure to a numpy array
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())

        # Convert the image to a torch tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        # Log the figure as an image in TensorBoard
        tb_logger["writer"].add_image(
            f"Layer {layer_idx + 1}/Attention Weights", image_tensor, global_step=step
        )
        plt.close(fig)


def log_values(
    model,
    cost,
    grad_norms,
    epoch,
    batch_id,
    step,
    log_likelihood,
    reinforce_loss,
    bl_loss,
    tb_logger,
    opts,
):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print(
        "epoch: {}, train_batch_id: {}, avg_cost: {}".format(epoch, batch_id, avg_cost)
    )

    print("grad_norm: {}, clipped: {}".format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger["logger"].log_value("avg_cost", avg_cost, step)

        tb_logger["logger"].log_value("actor_loss", reinforce_loss.item(), step)
        tb_logger["logger"].log_value("negative_log_likelihood", -log_likelihood.mean().item(), step)

        tb_logger["logger"].log_value("grad_norm", grad_norms[0], step)
        tb_logger["logger"].log_value("grad_norm_clipped", grad_norms_clipped[0], step)

        if opts.baseline == "critic":
            tb_logger["logger"].log_value("critic_loss", bl_loss.item(), step)
            tb_logger["logger"].log_value("critic_grad_norm", grad_norms[1], step)
            tb_logger["logger"].log_value(
                "critic_grad_norm_clipped", grad_norms_clipped[1], step
            )

        if opts.display_graphs is not None:
            # Log the weights for each encoder layer
            model_ = get_inner_model(model)
            for layer_idx in range(model_.n_encode_layers):
                encoder_layer = model_.encoder.layers[layer_idx]
                for name, param in encoder_layer.named_parameters():
                    tb_logger["writer"].add_histogram(
                        f"Encoder Layer {layer_idx + 1}/{name}",
                        param.clone().cpu().data.numpy(),
                        epoch,
                    )

            # Log the weights of the model
            for name, param in model_.named_parameters():
                if "encoder" not in name:
                    tb_logger["writer"].add_histogram(
                        f"Parameter: {name}", param.clone().cpu().data.numpy(), epoch
                    )

            encoder_distance = {
                "input_distance": (
                    model_.encoder_data["input"][0, :, None, :]
                    - model_.encoder_data["input"][0, None, :, :]
                ).norm(p=2, dim=-1),
                "embedding_distance": (
                    model_.encoder_data["embeddings"][0, :, None, :]
                    - model_.encoder_data["embeddings"][0, None, :, :]
                ).norm(p=2, dim=-1),
            }

            plot_encoder_data(encoder_distance, tb_logger, epoch)
            plot_attention_weights(model_, tb_logger, epoch)

    if opts.hyperparameter_tuning:
        tb_logger["ray"].add_scalar("Average Cost", avg_cost, epoch)
        tb_logger["ray"].add_scalar("Actor Loss", reinforce_loss.item(), epoch)

        tb_logger["ray"].add_scalar("Grad Norm", grad_norms[0], epoch)
        tb_logger["ray"].add_scalar("Grad Norm Clipped", grad_norms_clipped[0], epoch)

        model_ = get_inner_model(model)
        # Log the weights of the model
        for name, param in model_.named_parameters():
            tb_logger["ray"].add_histogram(
                f"Parameter: {name}", param.clone().cpu().data.numpy(), epoch
            )
