
!pip install transformers[torch]
!pip install -q datasets transformers
!pip install peft

model_checkpoint = "google/vit-base-patch16-224"

from datasets import load_dataset

# Load the dataset
dataset = load_dataset("STEM-AI-mtl/City_map", split="train")

from transformers import AutoImageProcessor

image_processor  = AutoImageProcessor.from_pretrained(model_checkpoint)
image_processor

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
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

from datasets import Dataset
import PIL

clean_examples = {'image': [], 'text': []}
skipped_images = 0

for i in range(len(dataset)):
    try:
        example = dataset[i]
        image = example['image']
        clean_examples['image'].append(image)
        clean_examples['text'].append(example['text'])
    except PIL.UnidentifiedImageError:

        print(f"Error processing image {i}: UnidentifiedImageError")
        skipped_images += 1
        continue
    except Exception as e:

        print(f"Error processing image {i}: {e}")
        skipped_images += 1
        continue


clean_dataset = Dataset.from_dict(clean_examples)

print(f"Number of skipped images: {skipped_images}")

splits = clean_dataset.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

unique_labels = set(clean_dataset['text'])

label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}

unique_labels = set(clean_dataset['text'])

# Create a label map
label_map = {label: i for i, label in enumerate(unique_labels)}

import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])

    # Convert string labels to numerical labels
    labels = [label_map[example["text"]] for example in examples]
    labels = torch.tensor(labels, dtype=torch.long)

    return {"pixel_values": pixel_values, "labels": labels}

from datasets import load_metric

metric = load_metric("accuracy")

import numpy as np
# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

#!pip install accelerate -U

#!pip install transformers[torch]

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
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
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
    push_to_hub=False,
    label_names=["labels"],
)

trainer = Trainer(
    lora_model,
    args,
    train_dataset=train_ds,
    eval_dataset = val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
train_results = trainer.train()

repo_name = f"STEM-AI-mtl/City_map-{model_name}"
lora_model.push_to_hub(repo_name)

repo_name = f"STEM-AI-mtl/City_map-{model_name}"
lora_model.push_to_hub(repo_name)

