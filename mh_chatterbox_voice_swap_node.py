import os
import torch
import torchaudio
import tempfile # Moved import to top level

# Adjusted relative import for ChatterboxVC
from .local_chatterbox.chatterbox.vc import ChatterboxVC
# Adjusted relative import for AudioNodeBase
from .modules.audio_base import AudioNodeBase
from comfy.utils import ProgressBar

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
        # tempfile imported at top
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
        converted_wav = None # Initialize converted_wav
        audio_data = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 16000} # Initialize with empty audio
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
            
            audio_data = {
                "waveform": converted_wav.unsqueeze(0),  # Add batch dimension
                "sample_rate": vc_model.sr
            }
            message += f"\nVoice converted successfully"

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
                         torch.cuda.empty_cache()
                    if torch.backends.mps.is_available():
                         torch.mps.empty_cache()

                message += f"\nLoading VC model on {device}..."
                pbar.update_absolute(10)
                vc_model = ChatterboxVC.from_pretrained(device=device)
                pbar.update_absolute(50)

                converted_wav = vc_model.generate(
                    audio=input_audio_path,
                    target_voice_path=target_voice_path,
                )
                pbar.update_absolute(90)
                audio_data = {
                    "waveform": converted_wav.unsqueeze(0),
                    "sample_rate": vc_model.sr
                }
                message += f"\nVoice converted successfully after fallback."
            else:
                message += f"\nError during VC: {str(e)}"
                audio_data = input_audio # Return original on other errors
        except Exception as e:
             message += f"\nAn unexpected error occurred during VC: {str(e)}"
             # audio_data remains the empty placeholder
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

            if converted_wav is not None: # Successful conversion or fallback
                if keep_model_loaded:
                    MH_ChatterboxVoiceSwap._vc_model = vc_model
                    MH_ChatterboxVoiceSwap._vc_device = device
                elif MH_ChatterboxVoiceSwap._vc_model is not None:
                    message += "\nUnloading VC model as keep_model_loaded is False or conversion was successful without keeping."
                    MH_ChatterboxVoiceSwap._vc_model = None
                    MH_ChatterboxVoiceSwap._vc_device = None
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    if torch.backends.mps.is_available(): torch.mps.empty_cache()
            elif MH_ChatterboxVoiceSwap._vc_model is not None: # Error, but model was loaded
                message += "\nUnloading VC model due to error or keep_model_loaded is False."
                MH_ChatterboxVoiceSwap._vc_model = None
                MH_ChatterboxVoiceSwap._vc_device = None
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                if torch.backends.mps.is_available(): torch.mps.empty_cache()
        
        pbar.update_absolute(100)
        return (audio_data, message)