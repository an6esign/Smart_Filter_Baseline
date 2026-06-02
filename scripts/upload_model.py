from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="app/models/rubert_two_head_fear",
    repo_id="an9y/mks_rubert",
    repo_type="model"
)

print("uploaded")