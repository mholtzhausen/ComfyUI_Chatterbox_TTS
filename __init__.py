# This is the main __init__.py for the ComfyUI_Chatterbox_TTS custom node package.

# Import mappings from the Chatterbox sub-package
from .Chatterbox import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# The following print statement is for debugging and can be removed later.
print("ComfyUI_Chatterbox_TTS custom node package initialized (root __init__.py).")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']