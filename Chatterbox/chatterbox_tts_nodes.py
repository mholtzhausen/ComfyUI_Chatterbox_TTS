import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional

# Import directly from the chatterbox package
from ..local_chatterbox.chatterbox.tts import ChatterboxTTS
from ..local_chatterbox.chatterbox.vc import ChatterboxVC

from comfy.utils import ProgressBar

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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_file_path": ("STRING", {
                    "multiline": False,
                    "default": "./audiobook_content.txt"
                }),
                "max_words_per_section": ("INT", {
                    "default": 250,
                    "min": 50,
                    "max": 1000,
                    "step": 10
                }),
            }
        }

    RETURN_TYPES = (["STRING"],) # ComfyUI expects a list of types, even for a single list output
    RETURN_NAMES = ("text_sections",)
    FUNCTION = "process_audiobook"
    OUTPUT_NODE = False
    CATEGORY = "MH/Chatterbox TTS"

    def process_audiobook(self, text_file_path: str, max_words_per_section: int):
        """
        Processes a text file into sections for an audiobook.

        Args:
            text_file_path: Path to the text file.
            max_words_per_section: Maximum number of words per section.

        Returns:
            A tuple containing a list of text sections.
        """
        sections = []
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            words = content.split()
            current_section_words = []
            for word in words:
                current_section_words.append(word)
                if len(current_section_words) >= max_words_per_section:
                    sections.append(" ".join(current_section_words))
                    current_section_words = []
            
            # Add any remaining words as the last section
            if current_section_words:
                sections.append(" ".join(current_section_words))

        except FileNotFoundError:
            print(f"Error: File not found at {text_file_path}")
            # Return a tuple with an empty list, as per ComfyUI's expected output format
            return ([],) 
        except Exception as e:
            print(f"An error occurred: {e}")
            return ([],)

        # ComfyUI expects the output to be a tuple, even if it's a single list.
        # The list itself should contain strings.
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
    "MH_AudiobookProcessor": "MH Audiobook Processor",
}