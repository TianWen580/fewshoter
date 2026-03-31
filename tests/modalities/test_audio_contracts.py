import importlib

torch = importlib.import_module("torch")
pytest = importlib.import_module("pytest")

modality_base = importlib.import_module("fewshoter.modalities.base")
modality_audio = importlib.import_module("fewshoter.modalities.audio")

EncoderContract = modality_base.EncoderContract
AudioEncoder = modality_audio.AudioEncoder
AudioEmbeddingMetadata = modality_audio.AudioEmbeddingMetadata
validate_audio_metadata = modality_audio.validate_audio_metadata


class DummyAudioEncoder:
    modality = "audio"

    def __init__(self):
        self.metadata = AudioEmbeddingMetadata(
            sample_rate_hz=32000,
            window_seconds=5.0,
            embedding_dim=8,
            encoder_name="perch2",
            checkpoint="placeholder",
            preprocessing_policy="perch2-default",
            provenance_tag="offline-placeholder",
        )
        self.embedding_dim = self.metadata.embedding_dim

    def preprocess_audio(self, audio):
        return torch.stack([torch.as_tensor(item, dtype=torch.float32) for item in audio], dim=0)

    def provenance(self):
        return {
            "encoder_name": self.metadata.encoder_name,
            "checkpoint": self.metadata.checkpoint,
            "preprocessing_policy": self.metadata.preprocessing_policy,
            "provenance_tag": self.metadata.provenance_tag,
        }

    def encode(self, audio):
        batch = self.preprocess_audio(audio)
        return {"global": batch.mean(dim=-1, keepdim=True).repeat(1, self.embedding_dim)}


def test_audio_encoder_contract_is_runtime_checkable():
    encoder = DummyAudioEncoder()
    assert isinstance(encoder, EncoderContract)
    assert isinstance(encoder, AudioEncoder)

    features = encoder.encode([torch.ones(4), torch.zeros(4)])
    assert "global" in features
    assert features["global"].shape == (2, encoder.embedding_dim)


def test_audio_metadata_validation_rejects_invalid_values():
    valid = AudioEmbeddingMetadata(
        sample_rate_hz=32000,
        window_seconds=5.0,
        embedding_dim=1536,
        encoder_name="perch2",
        checkpoint="ckpt-1",
        preprocessing_policy="perch2-default",
        provenance_tag="offline-placeholder",
    )
    validate_audio_metadata(valid)

    with pytest.raises(ValueError, match="sample_rate_hz"):
        validate_audio_metadata(
            AudioEmbeddingMetadata(
                sample_rate_hz=0,
                window_seconds=5.0,
                embedding_dim=1536,
                encoder_name="perch2",
                checkpoint="ckpt-1",
                preprocessing_policy="perch2-default",
                provenance_tag="offline-placeholder",
            )
        )
