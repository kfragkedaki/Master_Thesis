from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='logs')


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
        tb_logger.log_value('avg_cost', avg_cost, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)

        # Log the weights for each encoder layer
        for layer_idx in range(model.n_encode_layers):
            encoder_layer = model.embedder.layers._modules[str(layer_idx)]
            for name, param in encoder_layer.named_parameters():
                writer.add_histogram(f'Encoder Layer {layer_idx + 1}/{name}', param.clone().cpu().data.numpy(), epoch)

        # Log the weights of the model
        for name, param in model.named_parameters():
            if 'embedder' not in name:
                writer.add_histogram(f'Parameter: {name}', param.clone().cpu().data.numpy(), epoch)

