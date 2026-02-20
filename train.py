import logging

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils import logging_utils
from slime.utils.misc import should_run_periodic_action

_logger = logging.getLogger(__name__)


def _log_eval_metrics(args, metrics):
    """Log eval metrics from the primary process and materialize eval table rows."""
    if not isinstance(metrics, dict):
        return
    if "eval/step" not in metrics:
        return

    payload = dict(metrics)
    table_rows = payload.pop("_eval_table_rows", None)
    logging_utils.log(args, payload, step_key="eval/step")

    if not (args.use_wandb and isinstance(table_rows, list) and table_rows):
        return
    if not isinstance(table_rows[0], dict):
        return

    try:
        import wandb

        if wandb.run is None:
            return
        columns = list(table_rows[0].keys())
        table = wandb.Table(
            columns=columns,
            data=[[row.get(col) for col in columns] for row in table_rows],
        )
        wandb.log({"eval/sample_results": table, "eval/step": payload["eval/step"]})
    except Exception:
        _logger.debug("wandb.Table logging for eval/sample_results skipped", exc_info=True)


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        eval_metrics = ray.get(rollout_manager.eval.remote(rollout_id=0))
        _log_eval_metrics(args, eval_metrics)

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            eval_metrics = ray.get(rollout_manager.eval.remote(rollout_id))
            _log_eval_metrics(args, eval_metrics)

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        offload_train()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            eval_metrics = ray.get(rollout_manager.eval.remote(rollout_id))
            _log_eval_metrics(args, eval_metrics)

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
