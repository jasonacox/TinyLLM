#!/usr/bin/python3
"""
Model Selection Script for TinyLLM

This script will list all available models and allow you to select one to use. 
It will then create a symbolic link to the selected model and copy the service
file to the /etc/systemd/system directory.  It will then restart the tinyllm
service and wait for the port to be open before exiting.

Author: Jason A. Cox
5 January 2024
https://github.com/jasonacox/TinyLLM

"""
import os
import socket
import time
import subprocess

host = "localhost"
port = 8000

# Function to check if the port is open
def check_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

# Function to list chat formats and ask user to select
def select_chat_format(file_path='services/chatformats'):
    try:
        with open(file_path, 'r') as file:
            chat_formats = [line.strip() for line in file.readlines()]
            if not chat_formats:
                print("No chat formats found in the file.")
                return None
            print("Select a chat format:")
            for idx, chat_format in enumerate(chat_formats, start=1):
                print(f"{idx}. {chat_format}")
            choice = input("\nEnter the number of your choice (or press Enter for no change): ")
            if choice == '':
                print("No change selected.")
                return None
            try:
                choice = int(choice)
                if 1 <= choice <= len(chat_formats):
                    selected_chat_format = chat_formats[choice - 1]
                    print(f"Selected chat format: {selected_chat_format}")
                    return selected_chat_format
                else:
                    print("Invalid selection. Please enter a number within the valid range.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None

# Function to list files in the current directory
def list_files(directory):
    files = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".gguf"):
            files.append((filename, os.path.join(directory, filename)))
    return files

# Function to search and replace text in a file
def search_replace_in_file(file_path, search_text, replace_text):
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
        updated_content = file_content.replace(search_text, replace_text)
        with open(file_path, 'w') as file:
            file.write(updated_content)
        #print(f"Search and replace completed in '{file_path}'.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to get context size from user
def get_context_size():
    while True:
        try:
            context_size_str = input("Enter model context size [2048]: ")
            if not context_size_str:
                return "2048"
            context_size = int(context_size_str)
            if context_size > 0:
                return str(context_size)
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

# Main function
def main():
    directory = "."
    files = list_files(directory)
    command = "file tinyllm"
    result = subprocess.check_output(command, shell=True)
    current = result.decode().split("link to ")[1].strip()
    print(f"Current model: {current}\n")
    print("Change LLM - Available models:\n")
    for index, (filename, path) in enumerate(files):
        if path == current:
            print(f" *{index+1}. {filename}")
        else: 
            print(f"  {index+1}. {filename}")

    print("\nSelect model by entering its number (enter for no change): ",end='')
    choice = input()
    try:
        choice = int(choice)
    except:
        print(" - No change")
        return
    if choice not in range(1, len(files)+1):
        print("\nInvalid selection. Please enter a number between 1 and the total number of files.")
        return
    
    filename = files[choice-1][0]
    path = files[choice-1][1]
    print(f" - Selected {path} - filename {filename}")
    # Create sym link to selected model
    os.system(f"rm tinyllm")
    os.system(f"ln -s {path} tinyllm")
    # Create service file if it does not exist
    if not os.path.exists(f"services/tinyllm.service.{filename}"):
        print("\nNo service defined for this model - creating...\n")
        os.system(f"cp services/tinyllm.service.template services/tinyllm.service.{filename}")
        os.system(command)
        # As user for chat format
        chatformat = select_chat_format()
        if chatformat is not None:
            search_replace_in_file(f"services/tinyllm.service.{filename}","chatml",chatformat)
        # Ask user for context size
        contextsize = get_context_size()
        if int(contextsize) != 2048:
            search_replace_in_file(f"services/tinyllm.service.{filename}","2048",contextsize)

    # Copy service to system location
    os.system(f"grep ExecStart services/tinyllm.service.{filename}")
    os.system(f"sudo cp services/tinyllm.service.{filename} /etc/systemd/system/tinyllm.service")
    # Restart service
    print(f"\n - Restart tinyllm")
    os.system(f"sudo systemctl daemon-reload")
    os.system(f"sudo /etc/init.d/tinyllm restart")

    # Wait for the port to be open
    print(f"Waiting for port {port} to be up...")
    while not check_port():
        time.sleep(1)

    print("READY")
    print("")

if __name__ == "__main__":
    main()