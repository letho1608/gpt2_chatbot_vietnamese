# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 24 August 2023
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Setup device for computation (use GPU if available, otherwise use CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load fine-tuned model and tokenizer from the final directory
output_dir = 'OneChatbotGPT2Vi/final'
print(f"Loading fine-tuned GPT-2 model and tokenizer from {output_dir}...")
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
    sample_output = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, max_length=256, do_sample=True, top_k=100, top_p=0.9, temperature=0.6)

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
