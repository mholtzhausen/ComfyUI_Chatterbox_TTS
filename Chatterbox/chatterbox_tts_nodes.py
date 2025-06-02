import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import re

# Import directly from the chatterbox package
from ..local_chatterbox.chatterbox.tts import ChatterboxTTS
from ..local_chatterbox.chatterbox.vc import ChatterboxVC

from comfy.utils import ProgressBar

class AudiobookSection:
    def __init__(self, text: str, narrator_details: Dict, characters_details: Dict, index: int, filename: str):
        self.text: str = text
        self.narrator: Dict = narrator_details
        self.characters: Dict = characters_details
        self.index: int = index
        self.filename: str = filename

# Monkey patch torch.load to use MPS or CPU if map_location is not specified
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        # Determine the appropriate device (MPS for Mac, else CPU)
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        kwargs['map_location'] = torch.device(device)
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load


class AudioNodeBase:
    """Base class for audio nodes with common utilities."""
    
    @staticmethod
    def create_empty_tensor(audio, frame_rate, height, width, channels=None):
        """Create an empty tensor with dimensions based on audio duration."""
        audio_duration = audio['waveform'].shape[-1] / audio['sample_rate']
        num_frames = int(audio_duration * frame_rate)
        if channels is None:
            return torch.zeros((num_frames, height, width), dtype=torch.float32)
        else:
            return torch.zeros((num_frames, height, width, channels), dtype=torch.float32)

# Text-to-Speech node
class MH_ChatterboxTTS(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Text-to-Speech functionality.
    """
    _tts_model = None
    _tts_device = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "exaggeration": ("FLOAT", {"default": 0.7, "min": 0.25, "max": 2.0, "step": 0.05}),
                "cfg_weight": ("FLOAT", {"default": 0.3, "min": 0.2, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.05, "max": 5.0, "step": 0.05}),
            },
            "optional": {
                "audio_prompt": ("AUDIO",),
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "generate_speech"
    CATEGORY = "MH/Chatterbox TTS"
    
    def generate_speech(self, text, exaggeration, cfg_weight, temperature, audio_prompt=None, use_cpu=False, keep_model_loaded=False):
        """
        Generate speech from text.
        
        Args:
            text: The text to convert to speech.
            exaggeration: Controls emotion intensity (0.25-2.0).
            cfg_weight: Controls pace/classifier-free guidance (0.2-1.0).
            temperature: Controls randomness in generation (0.05-5.0).
            audio_prompt: AUDIO object containing the reference voice for TTS voice cloning.
            use_cpu: If True, forces CPU usage even if CUDA is available.
            keep_model_loaded: If True, keeps the model loaded in memory after generation.
            
        Returns:
            Tuple of (audio, message)
        """
        # Determine device to use
        device = "cpu" if use_cpu else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        if use_cpu:
            message = "Using CPU for inference (GPU disabled)"
        elif torch.backends.mps.is_available() and device == "mps":
             message = "Using MPS (Mac GPU) for inference"
        elif torch.cuda.is_available() and device == "cuda":
             message = "Using CUDA (NVIDIA GPU) for inference"
        else:
            message = f"Using {device} for inference" # Should be CPU if no GPU found
        
        # Create temporary files for any audio inputs
        import tempfile
        temp_files = []
        
        # Create a temporary file for the audio prompt if provided
        audio_prompt_path = None
        if audio_prompt is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_prompt:
                    audio_prompt_path = temp_prompt.name
                    temp_files.append(audio_prompt_path)
                
                # Save the audio prompt to the temporary file
                prompt_waveform = audio_prompt['waveform'].squeeze(0)
                torchaudio.save(audio_prompt_path, prompt_waveform, audio_prompt['sample_rate'])
                message += f"\nUsing provided audio prompt for voice cloning: {audio_prompt_path}"
                
                # Debug: Check if the file exists and has content
                if os.path.exists(audio_prompt_path):
                    file_size = os.path.getsize(audio_prompt_path)
                    message += f"\nAudio prompt file created successfully: {file_size} bytes"
                else:
                    message += f"\nWarning: Audio prompt file was not created properly"
            except Exception as e:
                message += f"\nError creating audio prompt file: {str(e)}"
                audio_prompt_path = None
        
        tts_model = None
        wav = None # Initialize wav to None
        audio_data = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 16000} # Initialize with empty audio
        pbar = ProgressBar(100) # Simple progress bar for overall process
        try:
            # Load the TTS model or reuse if loaded and device matches
            if MH_ChatterboxTTS._tts_model is not None and MH_ChatterboxTTS._tts_device == device:
                tts_model = MH_ChatterboxTTS._tts_model
                message += f"\nReusing loaded TTS model on {device}..."
            else:
                if MH_ChatterboxTTS._tts_model is not None:
                    message += f"\nUnloading previous TTS model (device mismatch or keep_model_loaded is False)..."
                    MH_ChatterboxTTS._tts_model = None
                    MH_ChatterboxTTS._tts_device = None
                    if torch.cuda.is_available():
                         torch.cuda.empty_cache() # Clear CUDA cache if possible
                    if torch.backends.mps.is_available():
                         torch.mps.empty_cache() # Clear MPS cache if possible


                message += f"\nLoading TTS model on {device}..."
                pbar.update_absolute(10) # Indicate model loading started
                tts_model = ChatterboxTTS.from_pretrained(device=device)
                pbar.update_absolute(50) # Indicate model loading finished

                if keep_model_loaded:
                    MH_ChatterboxTTS._tts_model = tts_model
                    MH_ChatterboxTTS._tts_device = device
                    message += "\nModel will be kept loaded in memory."
                else:
                    message += "\nModel will be unloaded after use."

            # Generate speech
            message += f"\nGenerating speech for: {text[:50]}..." if len(text) > 50 else f"\nGenerating speech for: {text}"
            if audio_prompt_path:
                message += f"\nUsing audio prompt: {audio_prompt_path}"
            
            pbar.update_absolute(60) # Indicate generation started
            wav = tts_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
            pbar.update_absolute(90) # Indicate generation finished
            
            audio_data = {
                "waveform": wav.unsqueeze(0),  # Add batch dimension
                "sample_rate": tts_model.sr
            }
            message += f"\nSpeech generated successfully"
            return (audio_data, message)
            
        except RuntimeError as e:
            # Check for CUDA or MPS errors and attempt fallback to CPU
            error_str = str(e)
            fallback_to_cpu = False
            if "CUDA" in error_str and device == "cuda":
                message += "\nCUDA error detected during TTS. Falling back to CPU..."
                fallback_to_cpu = True
            elif "MPS" in error_str and device == "mps":
                 message += "\nMPS error detected during TTS. Falling back to CPU..."
                 fallback_to_cpu = True

            if fallback_to_cpu:
                device = "cpu"
                # Unload previous model if it exists
                if MH_ChatterboxTTS._tts_model is not None:
                    message += f"\nUnloading previous TTS model..."
                    MH_ChatterboxTTS._tts_model = None
                    MH_ChatterboxTTS._tts_device = None
                    if torch.cuda.is_available():
                         torch.cuda.empty_cache() # Clear CUDA cache if possible
                    if torch.backends.mps.is_available():
                         torch.mps.empty_cache() # Clear MPS cache if possible


                message += f"\nLoading TTS model on {device}..."
                pbar.update_absolute(10) # Indicate model loading started (fallback)
                tts_model = ChatterboxTTS.from_pretrained(device=device)
                pbar.update_absolute(50) # Indicate model loading finished (fallback)
                # Note: keep_model_loaded logic is applied after successful generation
                # to avoid keeping a failed model loaded.

                wav = tts_model.generate(
                    text=text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
                pbar.update_absolute(90) # Indicate generation finished (fallback)
                audio_data = {
                    "waveform": wav.unsqueeze(0),  # Add batch dimension
                    "sample_rate": tts_model.sr
                }
                message += f"\nSpeech generated successfully after fallback."
                return (audio_data, message)
            else:
                message += f"\nError during TTS: {str(e)}"
                return (audio_data, message)
        except Exception as e:
             message += f"\nAn unexpected error occurred during TTS: {str(e)}"
             return (audio_data, message)
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            # If keep_model_loaded is False, ensure model is not stored
            # This is done here to ensure model is only kept if generation was successful
            if not keep_model_loaded and MH_ChatterboxTTS._tts_model is not None:
                 message += "\nUnloading TTS model as keep_model_loaded is False."
                 MH_ChatterboxTTS._tts_model = None
                 MH_ChatterboxTTS._tts_device = None
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache() # Clear CUDA cache if possible
                 if torch.backends.mps.is_available():
                     torch.mps.empty_cache() # Clear MPS cache if possible

        pbar.update_absolute(100) # Ensure progress bar completes on success or error
        return (audio_data, message) # Fallback return, should ideally not be reached


        # If generation was successful and keep_model_loaded is True, store the model
        if keep_model_loaded and tts_model is not None:
             MH_ChatterboxTTS._tts_model = tts_model
             MH_ChatterboxTTS._tts_device = device
             message += "\nModel will be kept loaded in memory."
        elif not keep_model_loaded and MH_ChatterboxTTS._tts_model is not None:
             # This case handles successful generation when keep_model_loaded was True previously
             # but is now False. Ensure the model is unloaded.
             message += "\nUnloading TTS model as keep_model_loaded is now False."
             MH_ChatterboxTTS._tts_model = None
             MH_ChatterboxTTS._tts_device = None
             if torch.cuda.is_available():
                 torch.cuda.empty_cache() # Clear CUDA cache if possible
             if torch.backends.mps.is_available():
                 torch.mps.empty_cache() # Clear MPS cache if possible


        # Create audio data structure for the output
        audio_data = {
            "waveform": wav.unsqueeze(0),  # Add batch dimension
            "sample_rate": tts_model.sr if tts_model else 16000 # Use default sample rate if model loading failed
        }
        
        message += f"\nSpeech generated successfully"
        pbar.update_absolute(100) # Ensure progress bar completes on success
        
        return (audio_data, message)

# Voice Conversion node
class MH_ChatterboxVoiceSwap(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Voice Conversion functionality.
    """
    _vc_model = None
    _vc_device = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_audio": ("AUDIO",),
                "target_voice": ("AUDIO",),
            },
            "optional": {
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "convert_voice"
    CATEGORY = "MH/Chatterbox TTS"
    
    def convert_voice(self, input_audio, target_voice, use_cpu=False, keep_model_loaded=False):
        """
        Convert the voice in an audio file to match a target voice.
        
        Args:
            input_audio: AUDIO object containing the audio to convert.
            target_voice: AUDIO object containing the target voice.
            use_cpu: If True, forces CPU usage even if CUDA is available.
            keep_model_loaded: If True, keeps the model loaded in memory after conversion.
            
        Returns:
            Tuple of (audio, message)
        """
        # Determine device to use
        device = "cpu" if use_cpu else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        if use_cpu:
            message = "Using CPU for inference (GPU disabled)"
        elif torch.backends.mps.is_available() and device == "mps":
             message = "Using MPS (Mac GPU) for inference"
        elif torch.cuda.is_available() and device == "cuda":
             message = "Using CUDA (NVIDIA GPU) for inference"
        else:
            message = f"Using {device} for inference" # Should be CPU if no GPU found
        
        # Create temporary files for the audio inputs
        import tempfile
        temp_files = []
        
        # Create a temporary file for the input audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
            input_audio_path = temp_input.name
            temp_files.append(input_audio_path)
        
        # Save the input audio to the temporary file
        input_waveform = input_audio['waveform'].squeeze(0)
        torchaudio.save(input_audio_path, input_waveform, input_audio['sample_rate'])
        
        # Create a temporary file for the target voice
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_target:
            target_voice_path = temp_target.name
            temp_files.append(target_voice_path)
        
        # Save the target voice to the temporary file
        target_waveform = target_voice['waveform'].squeeze(0)
        torchaudio.save(target_voice_path, target_waveform, target_voice['sample_rate'])
        
        vc_model = None
        pbar = ProgressBar(100) # Simple progress bar for overall process
        try:
            # Load the VC model or reuse if loaded and device matches
            if MH_ChatterboxVoiceSwap._vc_model is not None and MH_ChatterboxVoiceSwap._vc_device == device:
                vc_model = MH_ChatterboxVoiceSwap._vc_model
                message += f"\nReusing loaded VC model on {device}..."
            else:
                if MH_ChatterboxVoiceSwap._vc_model is not None:
                    message += f"\nUnloading previous VC model (device mismatch or keep_model_loaded is False)..."
                    MH_ChatterboxVoiceSwap._vc_model = None
                    MH_ChatterboxVoiceSwap._vc_device = None
                    if torch.cuda.is_available():
                         torch.cuda.empty_cache() # Clear CUDA cache if possible
                    if torch.backends.mps.is_available():
                         torch.mps.empty_cache() # Clear MPS cache if possible

                message += f"\nLoading VC model on {device}..."
                pbar.update_absolute(10) # Indicate model loading started
                vc_model = ChatterboxVC.from_pretrained(device=device)
                pbar.update_absolute(50) # Indicate model loading finished

                if keep_model_loaded:
                    MH_ChatterboxVoiceSwap._vc_model = vc_model
                    MH_ChatterboxVoiceSwap._vc_device = device
                    message += "\nModel will be kept loaded in memory."
                else:
                    message += "\nModel will be unloaded after use."

            # Convert voice
            message += f"\nConverting voice to match target voice"
            
            pbar.update_absolute(60) # Indicate conversion started
            converted_wav = vc_model.generate(
                audio=input_audio_path,
                target_voice_path=target_voice_path,
            )
            pbar.update_absolute(90) # Indicate conversion finished
            
        except RuntimeError as e:
            # Check for CUDA or MPS errors and attempt fallback to CPU
            error_str = str(e)
            fallback_to_cpu = False
            if "CUDA" in error_str and device == "cuda":
                message += "\nCUDA error detected during VC. Falling back to CPU..."
                fallback_to_cpu = True
            elif "MPS" in error_str and device == "mps":
                 message += "\nMPS error detected during VC. Falling back to CPU..."
                 fallback_to_cpu = True

            if fallback_to_cpu:
                device = "cpu"
                # Unload previous model if it exists
                if MH_ChatterboxVoiceSwap._vc_model is not None:
                    message += f"\nUnloading previous VC model..."
                    MH_ChatterboxVoiceSwap._vc_model = None
                    MH_ChatterboxVoiceSwap._vc_device = None
                    if torch.cuda.is_available():
                         torch.cuda.empty_cache() # Clear CUDA cache if possible
                    if torch.backends.mps.is_available():
                         torch.mps.empty_cache() # Clear MPS cache if possible

                message += f"\nLoading VC model on {device}..."
                pbar.update_absolute(10) # Indicate model loading started (fallback)
                vc_model = ChatterboxVC.from_pretrained(device=device)
                pbar.update_absolute(50) # Indicate model loading finished (fallback)
                # Note: keep_model_loaded logic is applied after successful generation
                # to avoid keeping a failed model loaded.

                converted_wav = vc_model.generate(
                    audio=input_audio_path,
                    target_voice_path=target_voice_path,
                )
                pbar.update_absolute(90) # Indicate conversion finished (fallback)
            else:
                # Re-raise if it's not a CUDA/MPS error or we're already on CPU
                message += f"\nError during VC: {str(e)}"
                # Return the original audio
                message += f"\nError: {str(e)}"
                pbar.update_absolute(100) # Ensure progress bar completes on error
                return (input_audio, message)
        except Exception as e:
             message += f"\nAn unexpected error occurred during VC: {str(e)}"
             empty_audio = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 16000}
             for temp_file in temp_files:
                 if os.path.exists(temp_file):
                     os.unlink(temp_file)
             pbar.update_absolute(100) # Ensure progress bar completes on error
             return (empty_audio, message)
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            # If keep_model_loaded is False, ensure model is not stored
            # This is done here to ensure model is only kept if generation was successful
            if not keep_model_loaded and MH_ChatterboxVoiceSwap._vc_model is not None:
                 message += "\nUnloading VC model as keep_model_loaded is False."
                 MH_ChatterboxVoiceSwap._vc_model = None
                 MH_ChatterboxVoiceSwap._vc_device = None
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache() # Clear CUDA cache if possible
                 if torch.backends.mps.is_available():
                     torch.mps.empty_cache() # Clear MPS cache if possible

        # If generation was successful and keep_model_loaded is True, store the model
        if keep_model_loaded and vc_model is not None:
             MH_ChatterboxVoiceSwap._vc_model = vc_model
             MH_ChatterboxVoiceSwap._vc_device = device
             message += "\nModel will be kept loaded in memory."
        elif not keep_model_loaded and MH_ChatterboxVoiceSwap._vc_model is not None:
             # This case handles successful generation when keep_model_loaded was True previously
             # but is now False. Ensure the model is unloaded.
             message += "\nUnloading VC model as keep_model_loaded is now False."
             MH_ChatterboxVoiceSwap._vc_model = None
             MH_ChatterboxVoiceSwap._vc_device = None
             if torch.cuda.is_available():
                 torch.cuda.empty_cache() # Clear CUDA cache if possible
             if torch.backends.mps.is_available():
                 torch.mps.empty_cache() # Clear MPS cache if possible

        # Create audio data structure for the output
        audio_data = {
            "waveform": converted_wav.unsqueeze(0),  # Add batch dimension
            "sample_rate": vc_model.sr if vc_model else 16000 # Use default sample rate if model loading failed
        }
        
        message += f"\nVoice converted successfully"
        pbar.update_absolute(100) # Ensure progress bar completes on success
        
        return (audio_data, message)

# Audiobook Processor Node
class MH_AudiobookProcessor:
    """
    ComfyUI node for processing text files into sections for an audiobook.
    """
    @classmethod
    def INPUT_TYPES(cls): # Use 'cls' as is conventional for classmethods
        return {
            "required": {
                "text_file_path": ("STRING", {
                    "multiline": False,
                    "default": "./audiobook.txt" # Updated default
                }),
                "max_words_per_section": ("INT", {
                    "default": 250,
                    "min": 50,
                    "max": 1000,
                    "step": 10
                }),
            }
        }

    RETURN_TYPES = ("AUDIOBOOK_SECTIONS",)
    RETURN_NAMES = ("sections",)
    FUNCTION = "process_audiobook"
    OUTPUT_NODE = False
    CATEGORY = "MH/Chatterbox TTS"

    def process_audiobook(self, text_file_path: str, max_words_per_section: int) -> tuple[List[AudiobookSection],]:
        sections: List[AudiobookSection] = []
        narrator_details: Dict = {}
        character_settings: Dict = {}
        current_section_index = 0
        
        attribute_parser_regex = re.compile(r"(\w+)=[\"']?([^\"']+)[\"']?")

        def parse_attrs(attr_string: str) -> Dict:
            attrs = {}
            if attr_string:
                for match in attribute_parser_regex.finditer(attr_string):
                    key, value = match.groups()
                    if value.replace('.', '', 1).isdigit(): # Basic check for number
                        if '.' in value: attrs[key] = float(value)
                        else: attrs[key] = int(value)
                    elif value.lower() == 'true': attrs[key] = True
                    elif value.lower() == 'false': attrs[key] = False
                    else: attrs[key] = value
            return attrs

        try:
            if not os.path.exists(text_file_path):
                print(f"Error: File not found at {text_file_path}")
                return ([],)
            with open(text_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e: # Catch other potential file errors
            print(f"Error reading file {text_file_path}: {str(e)}")
            return ([],)

        # Store segments of text with their associated speaker type and parameters
        # Each segment is {"speaker_type": "narrator" or "character", "name": "char_name" (if character), "params": dict_of_params, "text": "speakable_text"}
        speakable_segments = []
        
        # Regex definitions
        narrator_tag_regex = re.compile(r"<narrator\s+(.+?)\s*/>", re.IGNORECASE)
        set_tag_regex = re.compile(r"<set\s+name=[\"']?(\w+)[\"']?\s*(.*?)\s*/>", re.IGNORECASE)
        dialogue_tag_regex = re.compile(r"<(\w+)(?:\s+([^>]*?))?\s*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)

        # Initial position in the content string
        current_pos = 0

        # 1. Parse Narrator Tag (should be at the beginning)
        narrator_match = narrator_tag_regex.match(content) # Use match for beginning
        if narrator_match:
            narrator_attrs_str = narrator_match.group(1)
            narrator_details.update(parse_attrs(narrator_attrs_str))
            current_pos = narrator_match.end()
        else:
            # Default narrator if not specified, or handle as an error if required
            narrator_details = {"voice": "default_narrator_voice"}
            print("Warning: Narrator tag not found or not at the beginning. Using default narrator settings.")

        # 2. Find all other tags and interspersing text
        # Create a list of all tag occurrences to process them in order
        all_found_tags = []
        for match in set_tag_regex.finditer(content, current_pos):
            all_found_tags.append({'type': 'set', 'match': match, 'start': match.start(), 'end': match.end()})
        for match in dialogue_tag_regex.finditer(content, current_pos):
            all_found_tags.append({'type': 'dialogue', 'match': match, 'start': match.start(), 'end': match.end()})
        
        all_found_tags.sort(key=lambda x: x['start'])

        # Process text and tags in order
        for tag_info in all_found_tags:
            match_obj = tag_info['match']
            # Add narrated text before the current tag
            if match_obj.start() > current_pos:
                text_before_tag = content[current_pos:match_obj.start()].strip()
                if text_before_tag:
                    speakable_segments.append({
                        "speaker_type": "narrator",
                        "params": narrator_details.copy(), # Current global narrator settings
                        "text": text_before_tag
                    })
            
            # Process the tag itself
            if tag_info['type'] == 'set':
                char_name = match_obj.group(1)
                attrs_str = match_obj.group(2)
                char_attrs = parse_attrs(attrs_str)
                if char_name not in character_settings:
                    character_settings[char_name] = {}
                character_settings[char_name].update(char_attrs) # Update global character settings
            
            elif tag_info['type'] == 'dialogue':
                char_name = match_obj.group(1)
                attrs_str = match_obj.group(2) # Attributes specific to this dialogue instance
                dialogue_text = match_obj.group(3).strip().replace('"', '') # Clean quotes

                # Combine base character settings with dialogue-specific overrides
                effective_char_params = character_settings.get(char_name, {}).copy()
                if attrs_str:
                    dialogue_overrides = parse_attrs(attrs_str)
                    effective_char_params.update(dialogue_overrides)
                
                if dialogue_text:
                    speakable_segments.append({
                        "speaker_type": "character",
                        "name": char_name,
                        "params": effective_char_params, # These are the settings for THIS segment of dialogue
                        "text": dialogue_text
                    })
            current_pos = match_obj.end()

        # Add any remaining narrated text after the last tag
        if current_pos < len(content):
            remaining_text = content[current_pos:].strip()
            if remaining_text:
                speakable_segments.append({
                    "speaker_type": "narrator",
                    "params": narrator_details.copy(),
                    "text": remaining_text
                })

        # 3. Split speakable_segments into AudiobookSections
        current_section_text_parts = []
        current_word_count = 0
        
        # The narrator_details and character_settings for an AudiobookSection should be the
        # *defaults* active at the beginning of that section.
        # The actual text in AudiobookSection.text is clean.
        # The challenge of passing per-segment voice parameters for dialogue within a section
        # is deferred if AudiobookSection only holds defaults.
        # For this implementation, we'll store clean text.

        for segment in speakable_segments:
            segment_text_clean = segment["text"] # Already clean
            words_in_segment = len(segment_text_clean.split())

            if not segment_text_clean:
                continue

            # If current section + new segment > max_words, finalize current section
            if current_section_text_parts and (current_word_count + words_in_segment > max_words_per_section):
                sections.append(AudiobookSection(
                    text=" ".join(current_section_text_parts),
                    narrator_details=narrator_details.copy(), # Global defaults at section creation
                    characters_details={k: v.copy() for k, v in character_settings.items()}, # Global defaults
                    index=current_section_index,
                    filename=f"section_{current_section_index:03d}.wav"
                ))
                current_section_index += 1
                current_section_text_parts = []
                current_word_count = 0
            
            # If the segment itself is larger than max_words_per_section, split it
            if words_in_segment > max_words_per_section:
                words = segment_text_clean.split()
                temp_buffer = []
                for i, word in enumerate(words):
                    temp_buffer.append(word)
                    if len(temp_buffer) == max_words_per_section or i == len(words) - 1:
                        # If there was pending text, flush it first. This shouldn't happen if logic above is correct.
                        if current_section_text_parts: # Safeguard
                             sections.append(AudiobookSection(
                                text=" ".join(current_section_text_parts),
                                narrator_details=narrator_details.copy(),
                                characters_details={k: v.copy() for k, v in character_settings.items()},
                                index=current_section_index,
                                filename=f"section_{current_section_index:03d}.wav"))
                             current_section_index += 1
                             current_section_text_parts = []
                             current_word_count = 0

                        sections.append(AudiobookSection(
                            text=" ".join(temp_buffer),
                            narrator_details=narrator_details.copy(),
                            characters_details={k: v.copy() for k, v in character_settings.items()},
                            index=current_section_index,
                            filename=f"section_{current_section_index:03d}.wav"
                        ))
                        current_section_index += 1
                        temp_buffer = []
                # After splitting a large segment, current_section_text_parts should be empty.
            else: # Segment fits or can be added to current section
                current_section_text_parts.append(segment_text_clean)
                current_word_count += words_in_segment
                # If adding this segment makes the current section full or over
                if current_word_count >= max_words_per_section:
                    sections.append(AudiobookSection(
                        text=" ".join(current_section_text_parts),
                        narrator_details=narrator_details.copy(),
                        characters_details={k: v.copy() for k, v in character_settings.items()},
                        index=current_section_index,
                        filename=f"section_{current_section_index:03d}.wav"
                    ))
                    current_section_index += 1
                    current_section_text_parts = []
                    current_word_count = 0
        
        # Add any remaining text as the last section
        if current_section_text_parts:
            sections.append(AudiobookSection(
                text=" ".join(current_section_text_parts),
                narrator_details=narrator_details.copy(),
                characters_details={k: v.copy() for k, v in character_settings.items()},
                index=current_section_index,
                filename=f"section_{current_section_index:03d}.wav"
            ))
            # current_section_index += 1 # Not needed for the very last section

        return (sections,)
# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MH_ChatterboxTTS": MH_ChatterboxTTS,
    "MH_ChatterboxVoiceSwap": MH_ChatterboxVoiceSwap,
    "MH_AudiobookProcessor": MH_AudiobookProcessor,
}

# Display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MH_ChatterboxTTS": "[MH] Chatterbox TTS",
    "MH_ChatterboxVoiceSwap": "[MH] Chatterbox Voice Conversion",
    "MH_AudiobookProcessor": "[MH] Audiobook Processor", # Updated display name
}