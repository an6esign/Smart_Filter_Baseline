import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="an9y/mks_rubert",
    repo_type="model",
    local_dir="app/models/rubert_two_head_fear",
    token=os.getenv("HF_TOKEN")
)