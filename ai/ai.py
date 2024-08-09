import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import pickle

# Load the dataset
file_path = 'ai/data/expanded_combined_dataset.csv'
df = pd.read_csv(file_path)

# Assuming your dataset has columns like 'question' and 'category'
X = df['QUESTION']  # Features (the questions)
y = df['BT_LEVEL']  # Labels (the categories to classify)

# Convert labels to numerical format
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Create a custom Dataset class to handle the data
class QuestionClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels[idx]
        if torch.backends.mps.is_built():
            torch.device("mps")  # for mac use
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Create datasets for training and validation
train_dataset = QuestionClassificationDataset(X_train, y_train, tokenizer)
val_dataset = QuestionClassificationDataset(X_val, y_val, tokenizer)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  # Adjust as needed
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    use_mps_device=True,
)

# Initialize the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=None,  # Optional: Implement metrics if needed
)

# Train the model
trainer.train()

# Save the fine-tuned model to a pickle file
model_output_path = 'ai/model/fine_tuned_bert_model.pkl'
with open(model_output_path, 'wb') as f:
    pickle.dump(model, f)

# Save the label encoder for later use
label_encoder_output_path = 'ai/model/label_encoder.pkl'
with open(label_encoder_output_path, 'wb') as f:
    pickle.dump(label_encoder, f)
