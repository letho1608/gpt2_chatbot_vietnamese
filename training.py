import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader

# Setup device for computation (use GPU if available, otherwise use CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pretrained model and tokenizer
print("Loading GPT-2 model and tokenizer")
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Define optimizer and scheduler for learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
print("Optimizer and scheduler have been set up.")

# Define a custom Dataset class to handle data from the file
class QADataset(Dataset):
    def __init__(self, file_path, tokenizer, separator_token='(separator)', max_length=512):
        self.tokenizer = tokenizer
        self.separator_token = separator_token
        self.max_length = max_length
        self.examples = []
        # Read data from file and process question-answer pairs
        print(f"Reading data from file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Split question and answer based on separator_token
                if separator_token in line:
                    prompt, answer = line.split(separator_token, 1)
                    prompt = prompt.strip()
                    answer = answer.strip()

                    # Concatenate question and answer into one string and tokenize
                    input_text = f"{prompt} {answer}"
                    tokenized_input = tokenizer.encode(input_text, max_length=self.max_length, truncation=True, return_tensors='pt').squeeze()
                    self.examples.append(tokenized_input)
        print(f"Read {len(self.examples)} question-answer pairs.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Prepare data
print("Preparing data.")
dataset = QADataset('fb_messages.txt', tokenizer, separator_token='(separator)')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
num_samples = len(dataset)
print(f"Data prepared with {num_samples} examples.")

# Define the output directory for saving the model (same location as script)
output_dir = 'OneChatbotGPT2Vi'
os.makedirs(output_dir, exist_ok=True)

# Start fine-tuning the model
print("Starting to fine-tune the model.")
model.to(device)
model.train()

for epoch in range(10):
    epoch_loss = 0
    print(f"\nEpoch {epoch + 1}/10")
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch.to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Continuous update within each batch
        print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{num_samples}, Loss {loss.item():.3f}")

    # Summary after each epoch
    print(f"Epoch {epoch + 1}, Average Loss {epoch_loss / len(dataloader):.3f}")

    # Save the model after each epoch
    epoch_output_dir = os.path.join(output_dir, f'epoch_{epoch + 1}')
    os.makedirs(epoch_output_dir, exist_ok=True)
    print(f"Saving model after Epoch {epoch + 1} to directory: {epoch_output_dir}")
    tokenizer.save_pretrained(epoch_output_dir)
    model.save_pretrained(epoch_output_dir)

# Save the final model and tokenizer after all epochs are completed
final_output_dir = os.path.join(output_dir, 'final')
os.makedirs(final_output_dir, exist_ok=True)
print(f"Saving final model and tokenizer to directory: {final_output_dir}")
tokenizer.save_pretrained(final_output_dir)
model.save_pretrained(final_output_dir)

def generate_answer(question):
    print(f"Generating answer for question: {question}")
    # Encode the question using the tokenizer
    input_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors='pt').to(device)

    # Generate the answer using the model
    sample_output = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, max_length=256, do_sample=True, top_k=100, top_p=0.9, temperature=0.6)

    # Decode the answer using the tokenizer
    answer = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    return answer

# Example usage
question = 'Question: Xin chào'
response = generate_answer(question)
print(f"\nAnswer: {response}\n")
