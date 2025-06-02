# This file will initialize the Chatterbox custom node package.

# Import node classes from their new locations
from .nodes.mh_chatterbox_tts_node import MH_ChatterboxTTS
from .nodes.mh_chatterbox_voice_swap_node import MH_ChatterboxVoiceSwap
from .nodes.mh_audiobook_processor_node import MH_AudiobookProcessor
from .nodes.mh_read_text_file_node import MH_ReadTextFile

# Import utilities that might be needed or that patch behavior
from .modules import patch_utils # Ensures torch.load is patched if this __init__ is loaded

# Global mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MH_ChatterboxTTS": MH_ChatterboxTTS,
    "MH_ChatterboxVoiceSwap": MH_ChatterboxVoiceSwap,
    "MH_AudiobookProcessor": MH_AudiobookProcessor,
    "MH_ReadTextFile": MH_ReadTextFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MH_ChatterboxTTS": "[MH] Chatterbox TTS",
    "MH_ChatterboxVoiceSwap": "[MH] Chatterbox Voice Conversion",
    "MH_AudiobookProcessor": "[MH] Audiobook Processor",
    "MH_ReadTextFile": "[MH] Read Text File",
}

# The following print statement is for debugging and can be removed later.
print(f"Chatterbox custom node package initialized (Chatterbox/__init__.py). Nodes mapped.")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']