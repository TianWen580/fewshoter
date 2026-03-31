import importlib
from typing import Any, Dict, Iterable

torch = importlib.import_module("torch")

episode_module = importlib.import_module("fewshoter.core.episode")
evaluation_module = importlib.import_module("fewshoter.evaluation.base")
learner_module = importlib.import_module("fewshoter.learners.base")
modality_module = importlib.import_module("fewshoter.modalities.base")

Episode = episode_module.Episode
EvaluatorContract = evaluation_module.EvaluatorContract
Metrics = evaluation_module.Metrics
LearnerContract = learner_module.LearnerContract
EncoderContract = modality_module.EncoderContract
ImageEncoder = modality_module.ImageEncoder


class DummyImageEncoder:
    embedding_dim = 4
    modality = "image"

    def preprocess_images(self, images):
        return torch.stack([torch.as_tensor(image) for image in images])

    def encode(self, images) -> Dict[str, Any]:
        batch = self.preprocess_images(images).float()
        return {"global": batch.mean(dim=-1, keepdim=False)}


class DummyLearner:
    def __init__(self):
        self.fitted = False

    def fit(self, episodes: Iterable[Episode]) -> None:
        list(episodes)
        self.fitted = True

    def predict(self, embeddings: Any):
        return torch.zeros(embeddings.shape[0], dtype=torch.long)

    @classmethod
    def from_encoder(cls, encoder: EncoderContract, config: Any):
        assert encoder.embedding_dim >= 1
        return cls()


class DummyEvaluator:
    def evaluate(self, learner: LearnerContract, episodes: Iterable[Episode]) -> Metrics:
        list(episodes)
        _ = learner
        return Metrics(
            accuracy=1.0,
            confidence_interval=0.0,
            per_class_accuracy={"class_a": 1.0},
        )


def test_encoder_contracts_are_runtime_checkable():
    encoder = DummyImageEncoder()
    assert isinstance(encoder, EncoderContract)
    assert isinstance(encoder, ImageEncoder)
    features = encoder.encode([torch.ones(2), torch.zeros(2)])
    assert "global" in features


def test_learner_contract_is_implementable():
    encoder = DummyImageEncoder()
    learner = DummyLearner.from_encoder(encoder, config={})
    assert isinstance(learner, LearnerContract)
    learner.fit([])
    preds = learner.predict(torch.randn(3, 4))
    assert preds.shape[0] == 3


def test_evaluator_contract_is_implementable():
    evaluator = DummyEvaluator()
    learner = DummyLearner()
    metrics = evaluator.evaluate(learner, [])
    assert isinstance(evaluator, EvaluatorContract)
    assert metrics.accuracy == 1.0
    assert "class_a" in metrics.per_class_accuracy
