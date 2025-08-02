from huggingface_hub import upload_folder, login

# STEP 1: Login (only needed once per environment/session)
# You can remove this line if already logged in via terminal using `huggingface-cli login`
# login(token="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")  # Replace with your token or leave blank to prompt

# STEP 2: Upload folder to Hugging Face Model Hub
upload_folder(
    repo_id="manasivivek/tumordetection",  # Your HF repo name
    folder_path="t5_finetune_final",              # Local model folder name
    repo_type="model",
    commit_message="Upload final T5 medical fine-tuned model"
)

print("✅ Upload successful!")
