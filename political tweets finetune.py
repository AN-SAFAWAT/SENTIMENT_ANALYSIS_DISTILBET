import pandas as pd
import sklearn
import torch

# Load the dataset
df = pd.read_csv('/Users/leo_mac/Downloads/archive/ExtractedTweets.csv')


# Check if the data needs further cleaning
from transformers import DistilBertTokenizerFast
from torch.utils.data import Dataset, DataLoader

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


class PoliticalTweetsDataset(Dataset):
    def __init__(self, tweets, targets, tokenizer, max_len):
        self.tweets = tweets
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(target, dtype=torch.long)
        }


# You might need to map the 'Party' to a binary or numerical value
label_dict = {'Democrat': 0, 'Republican': 1}
df['label'] = df['Party'].map(label_dict)

# Split the data into train and validation sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df['Tweet'], df['label'], test_size=0.1, random_state=42)

# Create instances of the dataset
train_dataset = PoliticalTweetsDataset(X_train.to_numpy(), y_train.to_numpy(), tokenizer, max_len=128)
val_dataset = PoliticalTweetsDataset(X_val.to_numpy(), y_val.to_numpy(), tokenizer, max_len=128)

from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(label_dict)  # The number of output labels--2 for binary classification
)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
trainer.evaluate()


model.save_pretrained('/Users/leo_mac/Library/Application Support/JetBrains/PyCharmCE2022.3/scratches')
tokenizer.save_pretrained('/Users/leo_mac/Library/Application Support/JetBrains/PyCharmCE2022.3/scratches')


