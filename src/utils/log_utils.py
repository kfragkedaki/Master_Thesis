from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
import torch

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

logdir = "logs/plots/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=logdir)


def plot_encoder_data(figures, step=0, title="Heatmap Figures"):
    fig, axs = plt.subplots(1, len(figures), layout="constrained")

    # Create the heatmap
    img = None
    for index, (name, figure) in enumerate(figures.items()):
        img = axs[index].imshow(
            figure, cmap="coolwarm", interpolation="nearest"
        )
        axs[index].set_title(name + ' ' + str(step))

        # Add text annotations to show the heatmap values
        for i in range(figure.shape[0]):
            for j in range(figure.shape[1]):
                text = axs[index].text(j, i, f'{figure[i, j]:.2f}', ha='center', va='center', color='white', fontsize=6)

    fig.colorbar(img, shrink=0.6)

    # Convert the figure to a numpy array
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert the image to a torch tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()

    # Write the image to TensorBoard
    writer.add_image(title, image_tensor, global_step=step)

    plt.close(fig)


def plot_attention_weights(model, step=0):
    for layer_idx in range(model.n_encode_layers):
        attention_layer = model.embedder.layers[layer_idx][0].module
        # First instance of the batch
        att = attention_layer.get_attention_weights()[:, 0, :, :].detach().cpu().numpy()

        # Plot the attention weights
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Layer {layer_idx + 1} Attention Weights', fontsize=16)

        for head in range(attention_layer.n_heads):
            ax = axs[head // 4, head % 4]
            heatmap = ax.imshow(att[head], cmap='coolwarm', interpolation='nearest')

            # Set axis labels and title
            ax.set_xlabel('To Node', fontsize=12)
            ax.set_ylabel('From Node', fontsize=12)
            ax.set_title(f'Attention Head {head + 1})', fontsize=12)

            # Add text annotations to show the attention weight values
            for i in range(att[head].shape[0]):
                for j in range(att[head].shape[1]):
                    text = ax.text(j, i, f'{att[head][i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        # Add colorbar for reference
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position of the colorbar
        cbar = plt.colorbar(heatmap, cax=cbar_ax)
        cbar.set_label('Attention Weight', fontsize=12)

        # Convert the figure to a numpy array
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())

        # Convert the image to a torch tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        # Log the figure as an image in TensorBoard
        writer.add_image(f'Layer {layer_idx + 1}/Attention Weights', image_tensor, global_step=step)
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
        tb_logger.log_value("avg_cost", avg_cost, step)

        tb_logger.log_value("actor_loss", reinforce_loss.item(), step)
        tb_logger.log_value("nll", -log_likelihood.mean().item(), step)

        tb_logger.log_value("grad_norm", grad_norms[0], step)
        tb_logger.log_value("grad_norm_clipped", grad_norms_clipped[0], step)

        if opts.baseline == "critic":
            tb_logger.log_value("critic_loss", bl_loss.item(), step)
            tb_logger.log_value("critic_grad_norm", grad_norms[1], step)
            tb_logger.log_value("critic_grad_norm_clipped", grad_norms_clipped[1], step)

        # Log the weights for each encoder layer
        for layer_idx in range(model.n_encode_layers):
            encoder_layer = model.embedder.layers[layer_idx]
            for name, param in encoder_layer.named_parameters():
                writer.add_histogram(
                    f"Encoder Layer {layer_idx + 1}/{name}",
                    param.clone().cpu().data.numpy(),
                    epoch,
                )

        # Log the weights of the model
        for name, param in model.named_parameters():
            if "embedder" not in name:
                writer.add_histogram(
                    f"Parameter: {name}", param.clone().cpu().data.numpy(), epoch
                )

        encoder_distance = {
            'input_distance': (model.encoder_data['input'][0, :, None, :] - model.encoder_data['input'][0, None, :, :]).norm(p=2, dim=-1),
            'embedding_distance': (
                        model.encoder_data['embeddings'][0, :, None, :] - model.encoder_data['embeddings'][0, None, :, :]).norm(p=2,
                                                                                                                  dim=-1)
        }

        # Compute cosine similarity between embeddings
        # only the first instance in the batch_size
        encoder_cosine = {
            'input_cos': torch.from_numpy(
                cosine_similarity(model.encoder_data['input'][0].numpy())),
            'embeddings_cos': torch.from_numpy(
                cosine_similarity(model.encoder_data['embeddings'][0].numpy()))
        }

        plot_encoder_data(encoder_distance, step)
        # plot_heatmaps(encoder_cosine, step, title="Heatmap Figures 2")

        plot_attention_weights(model, step)


# # Create the heatmap
# img1 = axs[0].imshow(
#     input_distance.cpu().detach().numpy(), cmap="hot", interpolation="nearest"
# )
# axs[0].set_title(f"input_distance_np ${step}")
#
# # Create the heatmap
# img2 = axs[1].imshow(
#     embeddings_distance.cpu().detach().numpy(), cmap="hot", interpolation="nearest"
# )
# axs[1].set_title(f"embeddings_distance_np ${step}")
# fig.colorbar(img2, ax=axs[1], shrink=0.6)
#
# # Convert the figure to an image
# fig.canvas.draw()
# image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
# image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
# image = np.expand_dims(image, 0)
#
# # Convert the image to a TensorFlow tensor
# image_tensor = tf.image.convert_image_dtype(image, dtype=tf.uint8)
#
# # Write the image to TensorBoard
# with writer.as_default():
#     tf.summary.image(f"Heatmap Figures", image_tensor, step=step)


# tf.summary.image("Training data", plot_to_image(fig), step=0)

#
# # If you want to visualize the cosine similarity as a heatmap
# plt.figure(figsize=(10, 10))
# plt.imshow(cos_sim, cmap='hot', interpolation="nearest")
# plt.title("Cosine similarity heatmap of the embeddings")
# plt.colorbar()
# plt.show()

# # Use t-SNE to reduce dimensionality to 2D for visualization
# embeddings_first_graph = embeddings[0].cpu().detach().numpy()
# tsne = TSNE(n_components=2, perplexity=embeddings_first_graph.shape[0] - 1)
# embeddings_2d = tsne.fit_transform(embeddings_first_graph)
#
# # Plot the 2D embeddings
# plt.figure(figsize=(8, 6))
# plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
# plt.title("2D visualization of the embeddings using t-SNE")
# plt.show()
#
# plt.figure(figsize=(8, 6))
# plt.scatter(input[0][:, 0], input[0][:, 1])
# plt.title("2D visualization of the embeddings using t-SNE")
# plt.show()