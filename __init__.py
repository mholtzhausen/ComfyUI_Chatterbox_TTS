# This is the main __init__.py for the ComfyUI_Chatterbox_TTS custom node package.

# Import mappings from the Chatterbox sub-package
from .mh_chatterbox_tts_node import MH_ChatterboxTTS
from .mh_chatterbox_voice_swap_node import MH_ChatterboxVoiceSwap
from .mh_audiobook_processor_node import MH_AudiobookProcessor
from .mh_read_text_file_node import MH_ReadTextFile
from .mh_debug_node import MH_Debug

# The following print statement is for debugging and can be removed later.
print("ComfyUI_Chatterbox_TTS custom node package initialized (root __init__.py).")

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "MH_ChatterboxTTS": MH_ChatterboxTTS,
    "MH_ChatterboxVoiceSwap": MH_ChatterboxVoiceSwap,
    "MH_AudiobookProcessor": MH_AudiobookProcessor,
    "MH_ReadTextFile": MH_ReadTextFile,
    "MH_Debug": MH_Debug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MH_ChatterboxTTS": "[MH] Chatterbox TTS",
    "MH_ChatterboxVoiceSwap": "[MH] Chatterbox Voice Conversion",
    "MH_AudiobookProcessor": "[MH] Audiobook Processor",
    "MH_ReadTextFile": "[MH] Read Text File",
    "MH_Debug": "MH Debug Node",
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']