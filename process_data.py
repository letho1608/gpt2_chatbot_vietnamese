import os
import json
import re

# Converts facebook messages from multiple JSON files to one TXT
def load_facebook_data(directory):
    messages_list = []
    current_author = None
    combined_message = ""

    # Collect all subdirectories in both 'inbox' and 'archived_threads'
    subdirs = [os.path.join(directory, 'inbox', d) for d in os.listdir(os.path.join(directory, 'inbox')) if os.path.isdir(os.path.join(directory, 'inbox', d))]
    subdirs += [os.path.join(directory, 'archived_threads', d) for d in os.listdir(os.path.join(directory, 'archived_threads')) if os.path.isdir(os.path.join(directory, 'archived_threads', d))]

    for subdir in subdirs:
        # Collect all json files in the subdirectory and sort them by number in descending order
        files = [f for f in os.listdir(subdir) if f.endswith('.json')]
        files.sort(key=lambda x: int(re.search(r'(\d+)', x).group()), reverse=True)

        for file in files:
            file_path = os.path.join(subdir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                messages = data['messages']

                # Reverses the order (messages are stored latest first)
                messages.reverse()

                # Combines multiple messages in a row from the same person to 1 message
                for message in messages:
                    if 'content' in message:
                        author = message.get('sender_name')
                        content = message['content'].encode('latin1').decode('utf-8')

                        if author == current_author:
                            combined_message += " " + content
                        else:
                            if combined_message:
                                messages_list.append(combined_message)
                            combined_message = content
                            current_author = author

    if combined_message:
        messages_list.append(combined_message)
    return messages_list


def message_dump(directory, output_raw, separator_token, output_separated):
    # Prepare facebook json
    messages = load_facebook_data(directory)

    # Write raw messages
    with open(output_raw, 'w', encoding='utf-8') as f:
        for text in messages:
            f.write(text + '\n')

    # Read raw messages
    with open(output_raw, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Separate raw messages
    with open(output_separated, 'w', encoding='utf-8') as f:
        for i in range(0, len(lines) - 1, 2):
            prompt = lines[i].strip()
            answer = lines[i + 1].strip()
            combined_line = f"{prompt} {separator_token} {answer}\n"
            f.write(combined_line)


if __name__ == "__main__":
    message_dump(directory='fb/your_facebook_activity/messages',
                 output_raw='fb_messages_raw.txt',
                 output_separated='fb_messages.txt',
                 separator_token='(separator)')
