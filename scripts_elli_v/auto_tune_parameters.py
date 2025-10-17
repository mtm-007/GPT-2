import torch, time, itertools,statistics, pandas as pd
from torch.cuda import OutOfMemoryError
import torch.nn as nn
from torch.nn import functional as F
import mmap, random, time, wandb
# Import your model, data loader, and get_batch / PrefetchLoader definitions here
from Transformer_functions import Head, MultiHeadAttention, FeedForward, Block, GPTLanguagemodel
from data_batching_and_chunk_data import get_batch, PrefetchLoader

from profiled_training_run import model, optimizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- initialize wandb ---
wandb.init(project="gpt-autotune", name="gpu_param_search")

# === Search space ===
batch_sizes = [16, 32, 64, 128]
block_sizes = [64, 128, 256, 512]
prefetch_values = [2, 4, 8]
num_threads_values = [1, 2, 4]
amp_options = [False, True]

results = []

for batch_size, block_size, prefetch, num_threads, use_amp in itertools.product(
    batch_sizes, block_sizes, prefetch_values, num_threads_values, amp_options
):
    config = {
        "batch_size": batch_size,
        "block_size": block_size,
        "prefetch": prefetch,
        "num_threads": num_threads,
        "AMP": use_amp,
    }
    wandb.log({"status": "testing", **config})
    print(f"\nTesting: {config}")

    try:
        train_loader = PrefetchLoader(split="train", prefetch=prefetch, num_threads=num_threads)
        torch.cuda.empty_cache()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # === measure one training step ===
        # t0 = time.time()
        # xb, yb = get_batch("train", batch_size=batch_size, block_size=block_size)
        # xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        # torch.cuda.synchronize()
        # t1 = time.time()

        torch.cuda.synchronize()
        t0 = time.time()
        xb, yb = train_loader.next()  # preloaded async batch
        torch.cuda.synchronize()
        t1 = time.time()

        optimizer.zero_grad(set_to_none=True)
        try:
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits, loss = model(xb, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = model(xb, yb)
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            t2 = time.time()

            data_time = t1 - t0
            gpu_time = t2 - t1
            total_time = data_time + gpu_time

            print(f"✅ step {total_time:.3f}s (data={data_time:.3f}s, gpu={gpu_time:.3f}s)")
            result = (batch_size, block_size, prefetch, num_threads, use_amp, data_time, gpu_time, total_time)
            results.append(result)

            wandb.log({
                **config,
                "data_time": data_time,
                "gpu_time": gpu_time,
                "step_time": total_time,
                "status": "success"
            })

        except OutOfMemoryError:
            print("❌ CUDA OOM")
            torch.cuda.empty_cache()
            wandb.log({**config, "status": "OOM"})
            continue

    except Exception as e:
        print(f"⚠️ Error: {e}")
        torch.cuda.empty_cache()
        wandb.log({**config, "status": f"error: {e}"})
        continue

# === Save and summarize ===
if results:
    results.sort(key=lambda x: x[-1])
    df = pd.DataFrame(results, columns=[
        "batch_size", "block_size", "prefetch", "threads", "AMP",
        "data_time", "gpu_time", "step_time"
    ])
    df.to_csv("tuning_results.csv", index=False)

    print("\n=== Top configurations ===")
    print(df.head())

    # log all results to wandb
    wandb.log({"tuning_results": wandb.Table(dataframe=df)})

    # log the best config
    best = df.iloc[0].to_dict()
    wandb.summary.update(best)

    # save best config to JSON for reuse
    with open("best_config.json", "w") as f:
        json.dump(best, f, indent=4)

    print("\n✅ Best config saved to best_config.json")
else:
    print("No valid configurations found.")
    wandb.alert(title="AutoTuner", text="No valid configuration succeeded.")

wandb.finish()
