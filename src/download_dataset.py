import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files(
    "ebrahimelgazar/pixel-art",
    path="dataset",
    unzip=True,
)
