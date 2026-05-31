import pytest

from chak.attachment import ContentType, Image, MimeType, TXT, default_txt_reader, detect_content_type

pytestmark = pytest.mark.unit


def test_detect_content_type_for_url_data_uri_local_path_and_text(tmp_path):
    local_file = tmp_path / "sample.txt"
    local_file.write_text("hello", encoding="utf-8")

    assert detect_content_type("https://example.com/file.txt") == ContentType.URL
    assert detect_content_type("data:image/png;base64,abc") == ContentType.DATA_URI
    assert detect_content_type(str(local_file)) == ContentType.LOCAL_PATH
    assert detect_content_type("plain text") == ContentType.TEXT


def test_image_auto_detects_mime_type():
    assert Image("https://example.com/a.png").mime_type == MimeType.PNG
    assert Image("https://example.com/a.jpg").mime_type == MimeType.JPEG


def test_txt_reader_reads_plain_text_directly():
    attachment = TXT("hello world", reader=default_txt_reader)
    result = attachment.read()

    assert result.content == "hello world"
    assert result.meta["length"] == len("hello world")
