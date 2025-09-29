from transformers import pipeline
from utils import measure_time, log_call

class ModelInfoMixin:
    def model_info(self):
        return f"Model: {self._model_name}\nTask: {self._task}\n"

class LoggerMixin:
    def log(self, message):
        print(f"[MODEL LOG] {message}")

class AIModel(ModelInfoMixin, LoggerMixin):
    def __init__(self, model_name, task):
        self._model_name = model_name
        self._task = task
        self._pipeline = None

    @property
    def model_name(self):
        return self._model_name

    def load(self):
        self.log(f"Loading {self._model_name} for task {self._task}")
        self._pipeline = pipeline(self._task, model=self._model_name)

    def predict(self, input_data):
        raise NotImplementedError

class TextClassifier(AIModel):  # renamed to fit main.py
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        super().__init__(model_name, "text-classification")

    @measure_time
    @log_call
    def predict(self, text):
        if self._pipeline is None:
            self.load()
        result = self._pipeline(text)[0]
        return f"Label: {result['label']} (Confidence: {result['score']:.2f})"

class ImageClassifier(AIModel):  # renamed to fit main.py
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super().__init__(model_name, "image-classification")

    @measure_time
    @log_call
    def predict(self, image_path_or_pil):
        if self._pipeline is None:
            self.load()
        result = self._pipeline(image_path_or_pil)[0]
        return f"Label: {result['label']} (Confidence: {result['score']:.2f})"
