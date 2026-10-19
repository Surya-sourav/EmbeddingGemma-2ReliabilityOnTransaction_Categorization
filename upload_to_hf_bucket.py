"""Publish the notebook, README, data and results to a public Hugging Face Bucket.

    hf auth login                                         # once, with a WRITE token
    python upload_to_hf_bucket.py <your-username>/<bucket-name>

Run it from this folder after unzipping results_and_data.zip here and saving the executed notebook from Colab.
"""
import json
import shutil
import sys
from pathlib import Path

bucket = sys.argv[1]
notebook = "transactions_gemma2Embeddings_blindspot.ipynb"

# the notebook should be the one with outputs, saved from Colab
if not any(c.get("outputs") for c in json.load(open(notebook))["cells"]):
    sys.exit(f"{notebook} has no outputs - save the executed notebook from Colab first")

# copy just what we want to publish into a clean folder
stage = Path("_to_upload")
shutil.rmtree(stage, ignore_errors=True)
stage.mkdir()
for f in ["README.md", notebook, "requirements.txt", "upload_to_hf_bucket.py"]:
    shutil.copy(f, stage / f)
for d in ["data", "results"]:
    shutil.copytree(d, stage / d)

try:
    from huggingface_hub import bucket_info, create_bucket, list_bucket_tree, sync_bucket, update_bucket_settings
    from huggingface_hub.errors import BucketNotFoundError
except ImportError:
    # older huggingface_hub without Buckets: publish as a public dataset repo instead
    from huggingface_hub import create_repo, upload_folder
    create_repo(bucket, repo_type="dataset", private=False, exist_ok=True)
    upload_folder(repo_id=bucket, repo_type="dataset", folder_path=stage)
    sys.exit(f"Buckets not available, published as a dataset repo instead: https://huggingface.co/datasets/{bucket}")

# create the bucket as public, or make an existing one public, then upload
try:
    if bucket_info(bucket).private:
        update_bucket_settings(bucket, private=False)
except BucketNotFoundError:
    create_bucket(bucket, private=False)
sync_bucket(str(stage), f"hf://buckets/{bucket}")

n_files = sum(1 for _ in list_bucket_tree(bucket, recursive=True))
print(f"done: {n_files} entries in https://huggingface.co/buckets/{bucket}")
