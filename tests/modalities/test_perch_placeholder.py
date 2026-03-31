import importlib

torch = importlib.import_module("torch")
pytest = importlib.import_module("pytest")

audio_pipeline_module = importlib.import_module("fewshoter.perch_integration.audio_pipeline")
perch_encoder_module = importlib.import_module("fewshoter.perch_integration.perch_encoder")

AudioPipelineBoundary = audio_pipeline_module.AudioPipelineBoundary
AudioPreprocessingConfig = audio_pipeline_module.AudioPreprocessingConfig
AudioProvenance = audio_pipeline_module.AudioProvenance
PerchEncoderPlaceholder = perch_encoder_module.PerchEncoderPlaceholder


def _make_placeholder(*, sample_rate_hz=32000, window_seconds=5.0, embedding_dim=1536):
    pipeline = AudioPipelineBoundary(
        preprocessing=AudioPreprocessingConfig(
            sample_rate_hz=sample_rate_hz,
            window_seconds=window_seconds,
            channel_count=1,
            window_hop_seconds=5.0,
            policy_name="perch2-default",
        ),
        provenance=AudioProvenance(
            encoder_name="perch2",
            checkpoint="placeholder",
            provenance_tag="offline-placeholder",
            runtime_framework="none",
        ),
        embedding_dim=embedding_dim,
    )
    return PerchEncoderPlaceholder(pipeline=pipeline)


def test_perch_placeholder_encodes_without_runtime_dependency():
    encoder = _make_placeholder()
    output = encoder.encode([torch.randn(160000), torch.randn(160000)])

    assert encoder.modality == "audio"
    assert output["global"].shape == (2, 1536)
    assert encoder.provenance()["encoder_name"] == "perch2"
    assert "perch2-default" in encoder.pipeline.cache_key()


def test_perch_placeholder_invalid_metadata_rejected():
    with pytest.raises(ValueError, match="sample_rate_hz"):
        _make_placeholder(sample_rate_hz=16000)

    with pytest.raises(ValueError, match="window_seconds"):
        _make_placeholder(window_seconds=2.0)

    with pytest.raises(ValueError, match="embedding_dim"):
        _make_placeholder(embedding_dim=512)


def test_perch_placeholder_rejects_bad_embedding_shape_from_fake():
    pipeline = AudioPipelineBoundary(
        preprocessing=AudioPreprocessingConfig(sample_rate_hz=32000, window_seconds=5.0),
        provenance=AudioProvenance(),
        embedding_dim=16,
    )

    encoder = PerchEncoderPlaceholder(
        pipeline=pipeline,
        embedding_fn=lambda audio, dim: torch.ones(len(audio), dim + 1),
        expected_embedding_dim=16,
    )

    with pytest.raises(ValueError, match="dimension mismatch"):
        encoder.encode([torch.randn(32000)])
