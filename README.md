# GPT-2 Chatbot Vietnamese

## Purpose
This script demonstrates how to fine-tune a pre-trained GPT-2 model loaded from Hugging Face using a dataset of question-answer pairs extracted from individual conversations on Facebook in Vietnamese.

## Function
- Fine-tune the pre-trained GPT-2 model on a dataset consisting of question-answer pairs.
- The data set is created from personal Facebook chat logs in Vietnamese.

## Analysis
1. **Extract data**:
 - Download personal information from Facebook account.
 - Use Python to extract and pre-process conversations from downloaded Facebook data to create a refined dataset.

2. **Training**:
 - Script to refine the GPT-2 model using:
 - 1000 samples from the data set, or all
 - Complete data set over 10 epochs.
 - The model is saved after each epoch.

## Communicate
- **CPU device**:
 - Please note that training on CPU may result in long waiting times due to the computational intensity of model fine-tuning.

## Using
1. **Prepare data**:
 - Make sure your dataset of question-answer pairs is saved in `fb_messages.txt`.

2. **Run fine-tuning**:
 - Execute the script to start the tuning process.
 - The model will be saved in the `OneChatbotGPT2Vi` folder after each epoch and in the `final` subfolder after all epochs are completed.

3. **Create feedback**:
 - Use the refined model to generate answers to new questions using the provided code snippets.
