import torch
from huggingface_hub import hf_hub_download
from abc import ABC, abstractmethod
from ..core.audio_samples import AudioSamples

class ModelBase(ABC):
    @abstractmethod
    def load(self, use_gpu=True):
        pass

    @abstractmethod
    def forward(self, audio: AudioSamples):
        pass

class ECAPA2Model(ModelBase):
    def load(self, use_gpu=True):
        model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt')
        if use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.model = torch.jit.load(model_file, map_location=device)
        # Uncomment the following line if you want to use half precision
        # self.model.half()
        self.device = device
        return self.model, self.device

    def forward(self, audio: AudioSamples):
        # Ensure the audio data is on the same device as the model
        audio_tensor = audio.to_tensor().to(self.device)
        return self.model(audio_tensor)

class ModelFactory:
    @staticmethod
    def get_model(model_name):
        if model_name.upper() == "ECAPA2":
            return ECAPA2Model()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

class Extractor:
    def __init__(self, model_name, use_gpu=True):
        self.model_class = ModelFactory.get_model(model_name)
        self.model, self.device = self.model_class.load(use_gpu)

    def extract_embedding(self, audio: AudioSamples):
        """
        Extracts embedding from an Audio object.

        Parameters:
        audio (Audio): The Audio object to extract embedding from.

        Returns:
        torch.Tensor: The extracted embedding.
        """
        embedding = self.model_class.forward(audio)
        return embedding

    @property
    def is_using_gpu(self):
        return self.device.type == 'cuda'