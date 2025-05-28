from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

# config
BASE_MODEL_NAME = "DeepPavlov/rubert-base-cased"
LORA_ADAPTER_PATH = r"hw3\results\my_checkpoint"
TEXT_FILE_PATH = r"hw3\ozon_reviews.txt"
NUM_LABELS = 3 
LABEL_MAP = {0: "NEUTRAL", 1: "POSITIVE", 2: "NEGATIVE"}


def load_model_for_inference(base_model_name, lora_adapter_path, num_labels_for_model):
    print(f"Loading base model from '{base_model_name}'.")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels_for_model,
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model.eval() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded")
    return model, device

def predict_sentiment(text, model, tokenizer, device, label_map_dict):
    """Predict the sentiment of a single text。"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_label_id = torch.argmax(logits, dim=-1).item()
    return label_map_dict.get(predicted_label_id, "NA")


if __name__ == "__main__":
    print("Start")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    try:
        inference_model, inference_device = load_model_for_inference(
            BASE_MODEL_NAME, LORA_ADAPTER_PATH, NUM_LABELS
        )
    except Exception as e:
        print(f"Fail: {e}")
        print("Wrong Path")
        exit() 

    try:
        with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as f:
            print(f"\nRead File: {TEXT_FILE_PATH}\n")
            lines_processed_count = 0
            for i, line in enumerate(f):
                if lines_processed_count >= 30: 
                    break
                comment = line.strip()
                if comment: 
                    sentiment = predict_sentiment(
                        comment, inference_model, tokenizer, inference_device, LABEL_MAP
                    )
                    print(f"Comment {lines_processed_count + 1}: \"{comment}\"")
                    print(f"Sentiment: {sentiment}\n")
                    lines_processed_count += 1
            if lines_processed_count == 0:
                print("-------------------")

    except FileNotFoundError:
        print(f"Wrong Path {TEXT_FILE_PATH}")

    print("\nFinish。")
