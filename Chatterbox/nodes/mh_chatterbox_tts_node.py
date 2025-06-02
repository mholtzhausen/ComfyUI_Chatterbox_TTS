import os
import torch
import torchaudio
import tempfile # Moved import to top level

# Adjusted relative import for ChatterboxTTS
from ...local_chatterbox.chatterbox.tts import ChatterboxTTS
# Adjusted relative import for AudioNodeBase
from ..modules.audio_base import AudioNodeBase
from comfy.utils import ProgressBar

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
        # tempfile imported at top
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
            # Return was here, moved to after finally for consistency
            
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
                # Return was here, moved to after finally
            else:
                message += f"\nError during TTS: {str(e)}"
                # Return was here, moved to after finally
        except Exception as e:
             message += f"\nAn unexpected error occurred during TTS: {str(e)}"
             # Return was here, moved to after finally
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            
            # Model unloading logic based on success and keep_model_loaded
            if wav is not None: # Indicates successful generation or successful fallback
                if keep_model_loaded:
                    MH_ChatterboxTTS._tts_model = tts_model
                    MH_ChatterboxTTS._tts_device = device
                    # message += "\nModel will be kept loaded in memory." # Message already added or would be redundant
                elif MH_ChatterboxTTS._tts_model is not None: # Model was loaded, but not to be kept
                    message += "\nUnloading TTS model as keep_model_loaded is False or generation was successful without keeping."
                    MH_ChatterboxTTS._tts_model = None
                    MH_ChatterboxTTS._tts_device = None
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    if torch.backends.mps.is_available(): torch.mps.empty_cache()
            elif MH_ChatterboxTTS._tts_model is not None: # Error occurred, and a model was loaded
                message += "\nUnloading TTS model due to error or keep_model_loaded is False."
                MH_ChatterboxTTS._tts_model = None
                MH_ChatterboxTTS._tts_device = None
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                if torch.backends.mps.is_available(): torch.mps.empty_cache()

        pbar.update_absolute(100) # Ensure progress bar completes
        return (audio_data, message)