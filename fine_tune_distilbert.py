from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Load dataset
data = [
    {"text": "Women are too emotional to lead", "label": 1},
    {"text": "Both men and women are equally capable", "label": 0},
    {"text": "Some people are just naturally better at math", "label": 1},
    {"text": "Math skills depend on practice", "label": 0},
]

# Convert list of dictionaries to a dictionary of lists
data_dict = {"text": [item["text"] for item in data], "label": [item["label"] for item in data]}

# Create dataset
dataset = Dataset.from_dict(data_dict)

# Split dataset into train/test
train_test_split = dataset.train_test_split(test_size=0.2)
train_data = train_test_split["train"]
test_data = train_test_split["test"]

# Load pre-trained tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenize the data
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)

# Ensure necessary columns exist for training
train_data = train_data.remove_columns(["text"])
test_data = test_data.remove_columns(["text"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_distilbert",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./fine_tuned_distilbert/logs",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./fine_tuned_distilbert")
tokenizer.save_pretrained("./fine_tuned_distilbert")

print("Model and tokenizer fine-tuned and saved to './fine_tuned_distilbert'")
