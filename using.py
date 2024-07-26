import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Setup device for computation (use GPU if available, otherwise use CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Đường dẫn tuyệt đối đến thư mục cục bộ chứa mô hình và tokenizer đã fine-tuned
output_dir = 'OneChatbotGPT2Vi/final'
print(f"Loading fine-tuned GPT-2 model and tokenizer from {output_dir}...")

# Load mô hình và tokenizer từ thư mục cục bộ
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

# Set model to evaluation mode
model.to(device)
model.eval()

def generate_answer(question, model, tokenizer, device):
    print(f"Generating answer for question: {question}")
    # Encode the question using the tokenizer
    input_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors='pt').to(device)

    # Generate the answer using the model
    sample_output = model.generate(
        input_ids,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Decode the answer using the tokenizer
    answer = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    return answer

# Main loop for user input
while True:
    question = input("Question: ")
    if question.lower() == 'exit':
        print("Exiting the program.")
        break
    response = generate_answer(question, model, tokenizer, device)
    print(f"\nAnswer: {response}\n")
