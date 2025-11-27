from huggingface_hub import HfApi, upload_file
import os
import glob

# Your repo details
repo_id = "mtmx/gpt2_124M_1BTokens"

# Initialize API (will use your saved token automatically)
api = HfApi()

# Find all checkpoint files in current directory
checkpoint_files = sorted(glob.glob("model_*.pt"))

if not checkpoint_files:
    print("No model_*.pt files found in current directory")
    exit(1)

print(f"Found {len(checkpoint_files)} checkpoint files")
print(f"Uploading to: {repo_id}\n")

# Upload each checkpoint
for i, ckpt_file in enumerate(checkpoint_files, 1):
    file_size = os.path.getsize(ckpt_file) / (1024**3)  # GB
    print(f"[{i}/{len(checkpoint_files)}] Uploading {ckpt_file} ({file_size:.2f} GB)...")
    
    try:
        upload_file(
            path_or_fileobj=ckpt_file,
            path_in_repo=ckpt_file,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  ✓ Successfully uploaded {ckpt_file}\n")
    except Exception as e:
        print(f"  ✗ Failed to upload {ckpt_file}: {e}\n")

print("="*60)
print(f"✓ Upload complete!")
print(f"View your files at: https://huggingface.co/{repo_id}")
print("="*60)