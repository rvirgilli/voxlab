from ..core.audio_samples import AudioSamples
from .functions import break_into_chunks

class PreprocessingPipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, func, **kwargs):
        """
        Adds a preprocessing step to the pipeline.

        Parameters:
        func (function): The preprocessing function to add.
        kwargs (dict): Additional keyword arguments for the preprocessing function.
        """
        # Ensure break_into_chunks appears only once and as the last step
        if func == break_into_chunks:
            if any(step[0] == break_into_chunks for step in self.steps):
                raise ValueError("break_into_chunks can only be added once and must be the last step.")
            if self.steps and self.steps[-1][0] == break_into_chunks:
                raise ValueError("break_into_chunks can only be added once and must be the last step.")
            self.steps.append((func, kwargs))
        else:
            if self.steps and self.steps[-1][0] == break_into_chunks:
                raise ValueError("No steps can be added after break_into_chunks.")
            self.steps.append((func, kwargs))

    def process(self, audio: AudioSamples, **kwargs):
        for step, step_kwargs in self.steps:
            audio = step(audio, **{**step_kwargs, **kwargs.get(step.__name__, {})})
        return audio