import os

import pytest
import requests

import overpass
from overpass import (OverpassDownloadError, OverpassIncompleteError, download_with_retries,
                      get_read_timeout, validate_osm_xml)

VALID_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="Overpass API">
  <note>The data included in this document is from www.openstreetmap.org.</note>
  <node id="1" lat="43.6" lon="-79.4"/>
  <node id="2" lat="43.7" lon="-79.5"/>
  <way id="10">
    <nd ref="1"/>
    <nd ref="2"/>
    <tag k="highway" v="residential"/>
  </way>
</osm>
"""
TIMED_OUT_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="Overpass API">
  <node id="1" lat="43.6" lon="-79.4"/>
  <remark> runtime error: Query timed out in "query" at line 3 after 601 seconds. </remark>
</osm>
"""
HARMLESS_REMARK_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="Overpass API">
  <remark> runtime remark: Timeout is close to the limit. </remark>
  <node id="1" lat="43.6" lon="-79.4"/>
</osm>
"""
HTML_ERROR = b"""<?xml version="1.0" encoding="UTF-8"?>
<html><body><p>The server is probably too busy to handle your request.</p></body></html>
"""
QUERY = "[timeout:600][out:xml];way(1);out body;"


@pytest.fixture
def write_xml(tmp_path):
    def _write(content):
        path = tmp_path / "area.xml"
        path.write_bytes(content)
        return path
    return _write


def test_validate_accepts_complete_xml(write_xml):
    """Tests the validate function for a complete and valid XML"""
    validate_osm_xml(write_xml(VALID_XML))


def test_validate_accepts_harmless_remark(write_xml):
    """Tests the validate function for a xml with harmless remark """
    validate_osm_xml(write_xml(HARMLESS_REMARK_XML))


def test_validate_rejects_truncated_xml(write_xml):
    """Test the validate function for a xml that has been truncated due to a cut off"""
    with pytest.raises(OverpassIncompleteError, match="truncated"):
        validate_osm_xml(write_xml(VALID_XML[:200]))


def test_validate_rejects_runtime_error_remark(write_xml):
    """Test the validate function for a xml that has a remark for a timed out query"""
    with pytest.raises(OverpassIncompleteError, match="Query timed out"):
        validate_osm_xml(write_xml(TIMED_OUT_XML))


def test_validate_rejects_non_osm_root(write_xml):
    """Test the validate function that will reject a xml that doesn't have a osm root tag"""
    with pytest.raises(OverpassIncompleteError, match="root tag: <html>"):
        validate_osm_xml(write_xml(HTML_ERROR))


def test_read_timeout_outlasts_query_timeout():
    """Validate that the read timeout is always a minute longer than the query timeout set"""
    assert get_read_timeout(QUERY) == 660
    assert get_read_timeout("way(1);out;") == overpass.OVERPASS_DEFAULT_QUERY_TIMEOUT + 60


class FakeResponse:
    def __init__(self, chunks=(), status_code=200, fail_after_chunks=False):
        self.chunks = chunks
        self.status_code = status_code
        self.fail_after_chunks = fail_after_chunks
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error", response=self)

    def iter_content(self, chunk_size):
        yield from self.chunks
        if self.fail_after_chunks:
            raise requests.exceptions.ChunkedEncodingError("Connection broken")


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.urls = []

    def post(self, url, **kwargs):
        self.urls.append(url)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(overpass.time, "sleep", calls.append)
    return calls


def test_retries_after_connection_dropped_mid_download(tmp_path, sleeps):
    """Test that the process retries after connection gets dropped mid download""""
    output = str(tmp_path / "area.xml")
    session = FakeSession([
        FakeResponse([VALID_XML[:150]], fail_after_chunks=True),
        FakeResponse([VALID_XML[:150], VALID_XML[150:]]),
    ])
    assert download_with_retries(QUERY, output, session, ["https://a"], max_attempts=3, base_delay=1) == output
    validate_osm_xml(output)
    assert not os.path.exists(f"{output}.part")
    assert len(sleeps) == 1


def test_retries_after_runtime_error_and_rotates_urls(tmp_path, sleeps):
    """Test the process tries a different url if it maxes out retries and tries a different URL"""
    output = str(tmp_path / "area.xml")
    session = FakeSession([
        requests.exceptions.RetryError("too many 502 error responses"),
        FakeResponse([TIMED_OUT_XML]),
        FakeResponse([VALID_XML]),
    ])
    download_with_retries(QUERY, output, session, ["https://a", "https://b"], max_attempts=3, base_delay=1)
    assert session.urls == ["https://a", "https://b", "https://a"]
    validate_osm_xml(output)


def test_gives_up_after_max_attempts(tmp_path, sleeps):
    """Validate that process gives up after max number of attempts to download the xml is reached"""
    output = str(tmp_path / "area.xml")
    session = FakeSession([FakeResponse(status_code=504), FakeResponse([TIMED_OUT_XML])])
    with pytest.raises(OverpassDownloadError, match="after 2 attempts"):
        download_with_retries(QUERY, output, session, ["https://a"], max_attempts=2, base_delay=1)
    assert not os.path.exists(output)
    assert len(sleeps) == 1


def test_does_not_retry_bad_query(tmp_path, sleeps):
    """Validate that the process throws an error if it tries to download with a malformed query"""
    session = FakeSession([FakeResponse(status_code=400)])
    with pytest.raises(OverpassDownloadError, match="rejected the query"):
        download_with_retries(QUERY, str(tmp_path / "area.xml"), session, ["https://a"], max_attempts=3)
    assert sleeps == []
