# Chatterbox/nodes/mh_read_text_file_node.py
import os

class MH_ReadTextFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_file_path": ("STRING", {
                    "multiline": False,
                    "default": "./story.txt" # Or any other suitable default like "example.txt"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_content",)
    FUNCTION = "read_text_file"
    OUTPUT_NODE = False # This node is not a final output node
    CATEGORY = "MH/Utils" # Category for organization in ComfyUI

    def read_text_file(self, text_file_path: str) -> tuple[str]:
        # Validate text_file_path (must be a non-empty string)
        if not isinstance(text_file_path, str) or not text_file_path.strip():
            print(f"Error: MH_ReadTextFile - text_file_path is not a valid string or is empty.")
            return ("",) # Return empty string in a tuple for error

        try:
            # Check if the file exists
            if not os.path.exists(text_file_path):
                print(f"Error: MH_ReadTextFile - File not found at {text_file_path}")
                return ("",) # Return empty string in a tuple for error
            
            # Open and read the file
            with open(text_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return (content,) # Return content in a tuple
        except Exception as e:
            # Handle any other errors during file reading
            print(f"Error: MH_ReadTextFile - Error reading file {text_file_path}: {str(e)}")
            return ("",) # Return empty string in a tuple for error