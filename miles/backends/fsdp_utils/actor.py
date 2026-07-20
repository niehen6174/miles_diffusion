import logging
from argparse import Namespace
from collections import defaultdict

import ray
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

import miles.backends.fsdp_utils.configs.qwen_image  # noqa: F401 — register pipeline config
import miles.backends.fsdp_utils.configs.sd3  # noqa: F401 — register pipeline config
import miles.backends.fsdp_utils.configs.wan2_2  # noqa: F401 — register pipeline config
from miles.algorithms.base import TrainLossContext
from miles.algorithms.registry import load_algorithm
from miles.ray.train_actor import TrainRayActor
from miles.utils import tracking_utils, train_metric_utils
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.profile_utils import TrainProfiler
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils.train_data_utils import build_microbatch_schedule, validate_same_microbatch_counts_across_dp

from . import checkpoint
from .diffusion_update_weight_utils import (
    DiffusionUpdateWeightFromTensor,
    DiffusionUpdateWeightFromTensorLoRA,
    DiffusionUpdateWeightFromTensorLoRAIPC,
)
from .lr_scheduler import get_lr_scheduler
from .parallel import create_fsdp_parallel_state

logger = logging.getLogger(__name__)


def _enable_deterministic_training(args: Namespace) -> None:
    """Train-actor deterministic mode. NCCL/CUBLAS env is set at spawn (actor_group);
    here we set the torch-runtime knobs."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=False is required: SDPA's deterministic backward is gated on
    # !warnOnly (aten attention_backward.cu), so warn_only=True is a no-op on native.
    torch.use_deterministic_algorithms(True, warn_only=False)


class FSDPTrainRayActor(TrainRayActor):
    """FSDP training actor for diffusion algorithms.

    Loads only the DiT (transformer) from a diffusers pipeline, wraps it with
    FSDP, and delegates loss / train-example semantics to ``DiffusionAlgorithm``.
    """

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False) -> int:  # type: ignore[override]
        super().init(args, role, with_ref)

        if args.deterministic_mode:
            _enable_deterministic_training(args)

        self.parallel_state = create_fsdp_parallel_state(args)
        torch.manual_seed(args.seed)

        self.train_parallel_config = {
            "dp_size": self.parallel_state.dp_size,
        }

        if self.args.debug_rollout_only:
            return 0

        if self.args.offload_train and self.args.fsdp_cpu_offload:
            self.args.offload_train = False

        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        if self.args.start_rollout_id is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)

        self._master_dtype = _resolve_dtype(args.fsdp_master_dtype)
        self._forward_dtype = _resolve_dtype(args.diffusion_forward_dtype)

        from miles.utils.misc import load_function

        self.algorithm = load_algorithm(args)
        self.train_pipeline_config = load_function(args.train_pipeline_config_path)()
        self.train_pipeline_config.configure(args)
        self.model_backend = load_function(args.model_backend_path)(self.train_pipeline_config)
        if args.deterministic_mode:
            # flash-attn is opaque to torch's determinism flag; backends patch their own dispatch.
            self.model_backend.enable_deterministic_attention(args.fsdp_attention_backend)
        raw_models, self.scheduler = self.model_backend.load_models_and_scheduler(
            args, master_dtype=self._master_dtype
        )

        self.models: dict[str, torch.nn.Module] = {}
        for component, model in raw_models.items():
            # per raw component (wan2.2 has two transformers), before LoRA/FSDP wrap
            if args.fsdp_attention_backend is not None:
                self.model_backend.set_attention_backend(model, args.fsdp_attention_backend)

            if args.use_lora:
                model = apply_lora(model, args, self.train_pipeline_config)

            model.train()

            if args.gradient_checkpointing:
                self.model_backend.enable_gradient_checkpointing(model)

            model.to(torch.cuda.current_device())

            self.train_pipeline_config.preprocess_model_before_fsdp(model)

            model = apply_fsdp2(
                model,
                mesh=self.parallel_state.dp_mesh,
                cpu_offload=self.args.fsdp_cpu_offload,
                args=self.args,
                no_split_modules=self.model_backend.fsdp_no_split_modules(model),
            )
            self.models[component] = model
        # Force a sync to ensure sharding is complete and old memory is freed.
        torch.cuda.synchronize()
        clear_memory()

        if len(self.models) == 1:
            self.model = next(iter(self.models.values()))
        else:
            self.model = torch.nn.ModuleDict(self.models)

        from miles.utils.misc import load_function

        self.sde_backend = load_function(args.sde_step_backend_path)(
            self.scheduler,
            sde_timestep_divisor=self.train_pipeline_config.sde_timestep_divisor,
        )

        if args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                (p for p in self.model.parameters() if p.requires_grad),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

        # fp16 policy gradients are small enough to underflow without scaling.
        # ShardedGradScaler keeps the found_inf decision synchronized across
        # FSDP ranks; it is a no-op for bf16/fp32.
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        self.scaler = ShardedGradScaler(
            enabled=(self._forward_dtype == torch.float16),
        )

        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)
        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)

        # sglang-d now supports /update_weights_from_tensor (PR #20464).
        if self.args.debug_train_only:
            self.weight_updater = None
        elif self.args.use_lora and self.args.lora_ipc_weight_sync:
            self.weight_updater = DiffusionUpdateWeightFromTensorLoRAIPC(self.args, self.models)
        elif self.args.use_lora:
            self.weight_updater = DiffusionUpdateWeightFromTensorLoRA(self.args, self.models)
        else:
            self.weight_updater = DiffusionUpdateWeightFromTensor(self.args, self.models)

        checkpoint.finalize_load(self, checkpoint_payload)

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()

        return self.args.start_rollout_id

    def _get_parallel_config(self) -> dict:
        return {"dp_size": getattr(self.parallel_state, "dp_size", 1)}

    @timer
    def sleep(self) -> None:
        if not self.args.offload_train:
            return

        print_memory("before offload DiT")

        self.model.cpu()
        move_torch_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after sleep DiT")

    @timer
    def wake_up(self) -> None:
        if not self.args.offload_train:
            return

        self.model.cuda()
        move_torch_optimizer(self.optimizer, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up DiT")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:  # type: ignore[override]
        if self.args.save is None:
            return
        checkpoint.save(self, iteration=rollout_id)

    @timer
    def update_weights(self) -> None:  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.weight_updater is None:
            dist.barrier(group=get_gloo_group())
            return

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

        self.weight_updater.update_weights()
        clear_memory()

    def _gather_and_log_metrics(self, rollout_id: int, log_dict: dict[str, float], step: int) -> None:
        """Reduce per-rank scalars and log."""
        if "train/lr" not in log_dict and hasattr(self, "optimizer"):
            try:
                log_dict["train/lr"] = float(self.optimizer.param_groups[0]["lr"])
            except Exception:
                pass
        if self.parallel_state.dp_cp_rank == 0:
            dp_size = self.parallel_state.dp_cp_size
            gathered = [None] * dp_size
            dist.gather_object(
                log_dict,
                gathered,
                dst=self.parallel_state.dp_src_rank,
                group=self.parallel_state.dp_cp_group_gloo,
            )
            reduced = {k: sum(d[k] for d in gathered) / dp_size for k in log_dict}
            reduced["train/epoch"] = float(rollout_id)
            reduced["rollout/step"] = compute_rollout_step(self.args, rollout_id)
            reduced["train/step"] = float(step)
            tracking_utils.log(self.args, reduced, step_key="train/step")

            logger.info(
                f"[train step {int(step)}] rollout={rollout_id} "
                + " ".join(
                    f"{k}={v:.6e}"
                    for k, v in sorted(reduced.items())
                    if k not in ("train/epoch", "rollout/step", "train/step")
                )
            )
        else:
            dist.gather_object(
                log_dict,
                None,
                dst=self.parallel_state.dp_src_rank,
                group=self.parallel_state.dp_cp_group_gloo,
            )

    def train(self, rollout_id: int, rollout_data_ref) -> None:  # type: ignore[override]
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            rollout_data = ray.get(rollout_data_ref[self.parallel_state.dp_rank].inner)
            if self.args.debug_rollout_only:
                return
            self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
        )

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        """Train on ``rollout_data[train_data]`` via the configured ``DiffusionAlgorithm``.

        Optimizer windows are contiguous groups of train examples. Within a window, consecutive
        microbatches of size ``--micro-batch-size`` drive one forward+backward each; gradients
        scale as mean over all examples in the window (``loss_chunk / num_local_pairs``).
        """
        device = torch.cuda.current_device()

        train_pairs: list = rollout_data["train_data"]
        if not train_pairs:
            raise ValueError("rollout_data['train_data'] is empty")

        batch_errors = self.algorithm.validate_train_batch(train_pairs)
        if batch_errors:
            raise ValueError(f"Invalid train batch for {self.algorithm.name}: {batch_errors}")

        num_pairs = len(train_pairs)

        kl_beta = float(self.args.diffusion_kl_beta)
        if kl_beta > 0 and not self.args.use_lora:
            raise ValueError(
                "--diffusion-kl-beta currently requires --use-lora so the base model can be used as reference."
            )
        if kl_beta > 0 and not all(hasattr(m, "disable_adapter") for m in self.models.values()):
            raise RuntimeError("Diffusion KL requires PEFT models exposing disable_adapter() after FSDP wrapping.")

        loss_ctx = TrainLossContext(
            models=self.models,
            model=self.model,
            train_pipeline_config=self.train_pipeline_config,
            sde_backend=self.sde_backend,
            scheduler=self.scheduler,
            args=self.args,
            forward_dtype=self._forward_dtype,
            device=device,
        )
        self.algorithm.prepare_rollout_data(rollout_data, loss_ctx)

        # ------------- Micro-batch schedule -------------
        num_optim_steps_per_rollout = self.args.num_steps_per_rollout
        if num_pairs % num_optim_steps_per_rollout != 0:
            raise ValueError(
                f"num_pairs_shard={num_pairs} not divisible by " f"num_steps_per_rollout={num_optim_steps_per_rollout}"
            )
        num_pairs_per_optim_step = num_pairs // num_optim_steps_per_rollout
        micro_bs = self.args.micro_batch_size
        if micro_bs <= 0:
            raise ValueError(f"micro_batch_size must be positive, got {micro_bs}")
        microbatch_schedule = build_microbatch_schedule(
            num_pairs_per_optim_step=num_pairs_per_optim_step,
            num_optim_steps_per_rollout=num_optim_steps_per_rollout,
            micro_batch_size=micro_bs,
        )
        validate_same_microbatch_counts_across_dp(
            microbatch_schedule=microbatch_schedule,
            parallel_state=self.parallel_state,
        )

        # ------------- Forward / Backward -------------
        with timer("actor_train"):
            for microbatch_ranges in microbatch_schedule:
                self.optimizer.zero_grad(set_to_none=True)

                num_local_pairs = sum(pair_hi - pair_lo for pair_lo, pair_hi in microbatch_ranges)

                # LEGACY 2D parity: pad cond to the whole-window width. TODO: remove with legacy 2D path.
                legacy_pad_to_len = self._maybe_legacy_window_pad_len(train_pairs, microbatch_ranges)

                log_stats: dict[str, list[torch.Tensor]] = defaultdict(list)

                for pair_lo, pair_hi in microbatch_ranges:
                    chunk = train_pairs[pair_lo:pair_hi]
                    loss_sum = self.algorithm.compute_loss(
                        loss_ctx,
                        chunk,
                        log_stats=log_stats,
                        pad_to_len=legacy_pad_to_len,
                    )
                    if not self.args.debug_skip_optimizer_step:
                        # ShardedGradScaler keeps fp16 policy grads from underflowing
                        # (required for SD3.5 fp16 forward); no-op for bf16/fp32.
                        self.scaler.scale(loss_sum / float(num_local_pairs)).backward()

                self.prof.step(rollout_id=rollout_id)
                if not self.args.debug_skip_optimizer_step:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    if isinstance(grad_norm, DTensor):
                        # clip returns a lazily-reduced partial norm; materialize it,
                        # otherwise the logged metric leaks the local shard's value.
                        grad_norm = grad_norm.full_tensor()
                    log_stats["grad_norm"].append(grad_norm.detach())
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # Do mean over all ranks for now, may need to be updated for p99, max, etc.
                reduced = {f"train/{k}": torch.stack(v).mean().item() for k, v in log_stats.items()}
                self._gather_and_log_metrics(rollout_id, reduced, step=self.global_step)

    def _maybe_legacy_window_pad_len(self, train_pairs: list, microbatch_ranges: list) -> int | None:
        """LEGACY 2D parity: the whole-window max cond seq_len (like the legacy tile path), or
        None unless the legacy --micro-batch-size-sample>1 path is active. TODO: remove with it."""
        if self.args.micro_batch_size_sample is None or self.args.micro_batch_size_sample <= 1:
            return None
        conds = []
        for pair_lo, pair_hi in microbatch_ranges:
            for pair in train_pairs[pair_lo:pair_hi]:
                env = pair["denoising_env"]
                conds.append(env.pos_cond_kwargs)
                if env.neg_cond_kwargs is not None:
                    conds.append(env.neg_cond_kwargs)
        return self.train_pipeline_config.maybe_legacy_window_pad_len(conds)


@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    """ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py"""
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)

    torch.cuda.synchronize()


def _resolve_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def apply_lora(model: torch.nn.Module, args: Namespace, train_pipeline_config) -> None:
    """Apply PEFT LoRA to the model.

    Args:
        model: The model to apply LoRA to.
        args: Arguments containing LoRA settings.
        train_pipeline_config: The train pipeline config.
    """
    from peft import LoraConfig, get_peft_model

    # Per-model fallback when --lora-target-modules is unset (runtime inference: depends on loaded pipeline).
    targets = args.lora_target_modules or train_pipeline_config.lora_target_modules
    init_lora_weight = args.diffusion_init_lora_weight
    if init_lora_weight == "kaiming-uniform":
        init_lora_weight = True  # namely kaiming-uniform
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=targets,
            init_lora_weights=init_lora_weight,
        ),
    )
    if dist.get_rank() == 0:
        model.print_trainable_parameters()
    return model


def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None, no_split_modules=None):
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = no_split_modules if no_split_modules is not None else model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [module for name, module in model.named_modules() if module.__class__.__name__ in layer_cls_to_wrap]

    param_dtype = _resolve_dtype(args.diffusion_forward_dtype)
    reduce_dtype = _resolve_dtype(args.fsdp_reduce_dtype)
    logger.info(
        f"FSDP: wrapping {len(modules)} modules of type {layer_cls_to_wrap}, param_dtype={param_dtype}, reduce_dtype={reduce_dtype}"
    )

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    if args.gradient_checkpointing:
        # MixedPrecisionPolicy does not cast buffers; a buffer above param_dtype
        # makes the ckpt recompute dtype-diverge from the forward and abort.
        for module in model.modules():
            for name, buf in module.named_buffers(recurse=False):
                if buf.is_floating_point() and buf.dtype != param_dtype:
                    persistent = name not in module._non_persistent_buffers_set
                    module.register_buffer(name, buf.to(param_dtype), persistent=persistent)

    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)

    return model
