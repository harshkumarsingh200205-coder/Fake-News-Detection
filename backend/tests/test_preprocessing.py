from preprocessing import TextPreprocessor


def test_clean_text_removes_urls_and_html() -> None:
    preprocessor = TextPreprocessor()

    cleaned = preprocessor.clean_text(
        "<p>Breaking update</p> Visit https://example.com right now."
    )

    assert "<p>" not in cleaned
    assert "https://example.com" not in cleaned
    assert "Breaking update" in cleaned


def test_preprocess_normalizes_case_and_noise() -> None:
    preprocessor = TextPreprocessor()

    processed = preprocessor.preprocess(
        "BREAKING!!! Reuters said the economy grew by 3 percent."
    )

    assert processed == processed.lower()
    assert "reuters" not in processed
    assert "breaking" in processed
