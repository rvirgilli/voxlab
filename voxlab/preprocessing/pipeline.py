from pydub import AudioSegment
import os
from .functions import break_into_chunks

class PreprocessingPipeline:
    def __init__(self):
        """
        Initializes an empty preprocessing pipeline.
        """
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

    def load_audio(self, audio_path):
        """
        Loads the audio file based on its file type.

        Parameters:
        audio_path (str): The path to the audio file.

        Returns:
        AudioSegment: The loaded audio file.

        Raises:
        ValueError: If the file type is unsupported.
        """
        file_type = audio_path.split('.')[-1]
        if file_type == 'wav':
            audio = AudioSegment.from_wav(audio_path)
        elif file_type == 'opus':
            audio = AudioSegment.from_ogg(audio_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return audio

    def process(self, audio, **kwargs):
        """
        Processes the audio through all added preprocessing steps.

        Parameters:
        audio (AudioSegment): The audio to process.
        kwargs (dict): Additional keyword arguments for the preprocessing steps.

        Returns:
        AudioSegment or list: The processed audio or a list of processed audio chunks.
        """
        for step, step_kwargs in self.steps:
            audio = step(audio, **{**step_kwargs, **kwargs.get(step.__name__, {})})
        return audio

    def export(self, audio, export_path, format='wav'):
        """
        Exports the processed audio to a file or files.

        Parameters:
        audio (AudioSegment or list): The audio to export.
        export_path (str): The path to the folder to save the exported audio files.
        format (str): The format to save the file in (default is 'wav').

        Raises:
        ValueError: If the format is unsupported.
        """
        if format not in ['wav', 'mp3', 'ogg', 'flac']:
            raise ValueError(f"Unsupported export format: {format}")

        if isinstance(audio, list):
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            for i, chunk in enumerate(audio):
                chunk_file_path = os.path.join(export_path, f"{i:04d}.{format}")
                chunk.export(chunk_file_path, format=format)
        else:
            audio.export(export_path, format=format)
