import torch

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