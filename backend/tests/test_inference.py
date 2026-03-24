from inference import URLScraper


def test_url_scraper_accepts_valid_urls() -> None:
    assert URLScraper.is_valid_url("https://example.com/news/story")
    assert URLScraper.is_valid_url("http://127.0.0.1:8000/health")
    assert not URLScraper.is_valid_url("ftp://example.com")


def test_extract_text_removes_non_content_tags() -> None:
    html = """
    <html>
      <body>
        <header>Site header</header>
        <article>
          <p>First paragraph.</p>
          <p>Second paragraph.</p>
        </article>
        <script>ignore me</script>
      </body>
    </html>
    """

    extracted = URLScraper.extract_text(html)

    assert "First paragraph." in extracted
    assert "Second paragraph." in extracted
    assert "Site header" not in extracted
    assert "ignore me" not in extracted
