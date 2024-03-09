
model_checkpoint = "google/vit-base-patch16-224"

from datasets import load_dataset

# Load the dataset
dataset = load_dataset("STEM-AI-mtl/City_map", split="train")

# Extract unique text values as labels
unique_texts = set(dataset['text'])
text2id = {text: i for i, text in enumerate(unique_texts)}
id2text = {i: text for text, i in text2id.items()}

from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

#from PIL import Image
#import io

#from io import BytesIO

#def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    processed_images = []
    for image in example_batch["image"]:
        # Ensure image is read correctly into BytesIO
        image_bytes = BytesIO(image)
        image = Image.open(image_bytes).convert("RGB")
        processed_images.append(train_transforms(image))
    example_batch["pixel_values"] = processed_images
    return example_batch



splits = dataset.train_test_split(test_size=0.1)
train_ds = splits["train"]

train_ds.set_transform(preprocess_train)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(text2id),
    label2id=text2id,
    id2label=id2text,
    ignore_mismatched_sizes=True,
)

print_trainable_parameters(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

from transformers import TrainingArguments, Trainer

model_name = model_checkpoint.split("/")[-1]
batch_size = 128

args = TrainingArguments(
    f"{model_name}-finetuned-lora-City_map",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    label_names=["labels"],
)

import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])  # Adjust based on the actual key for labels
    return {"pixel_values": pixel_values, "labels": labels}


trainer = Trainer(
    lora_model,
    args,
    train_dataset=train_ds,
    tokenizer=image_processor,
    data_collator=collate_fn,
)
train_results = trainer.train()

repo_name = f"STEM-AI-mtl/City_map-{model_name}"
lora_model.push_to_hub(repo_name)