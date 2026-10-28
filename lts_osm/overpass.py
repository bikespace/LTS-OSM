"""Download OSM data from the Overpass API, retrying when the server fails or cuts a download short"""
import logging
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from string import Template

import requests
from requests.adapters import HTTPAdapter
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Comma-separated; later urls are fallbacks used when earlier ones keep failing
OVERPASS_API_URLS = [
    url.strip()
    for url in os.environ.get(
        "OVERPASS_API_URL",
        "https://overpass-api.de/api/interpreter,https://overpass.kumi.systems/api/interpreter",
    ).split(",")
    if url.strip()
]
OVERPASS_MAX_ATTEMPTS = int(os.environ.get("OVERPASS_MAX_ATTEMPTS", "6"))
OVERPASS_RETRY_BASE_DELAY = float(os.environ.get("OVERPASS_RETRY_BASE_DELAY", "60"))
OVERPASS_RETRY_MAX_DELAY = 600
OVERPASS_DEFAULT_QUERY_TIMEOUT = 180
OVERPASS_QUERY_TEMPLATE = Path("query/query_template.overpass")
USER_AGENT = "bikespace-LTS (https://github.com/bikespace/LTS-OSM)"


class OverpassIncompleteError(Exception):
    """The downloaded file is not complete OSM XML."""


class OverpassDownloadError(Exception):
    """Every attempt to download an area failed."""


def make_overpass_session() -> requests.Session:
    session = requests.Session()
    # Only a couple of quick retries here; download_with_retries handles longer outages
    retry = Retry(
        total=2,
        backoff_factor=2,
        status_forcelist={429, 502, 503, 504},
        respect_retry_after_header=True,
        allowed_methods={"POST"},
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def build_overpass_query(area_id: str) -> str:
    template_text = OVERPASS_QUERY_TEMPLATE.read_text()
    return Template(template_text).substitute(key="wikidata", value=area_id)


def get_read_timeout(query_str: str) -> int:
    # Overpass may send nothing until the query finishes, so outlast the server-side [timeout:]
    match = re.search(r"\[timeout:(\d+)\]", query_str)
    query_timeout = int(match.group(1)) if match else OVERPASS_DEFAULT_QUERY_TIMEOUT
    return query_timeout + 60


def validate_osm_xml(xml_path: str | Path) -> None:
    """Raise OverpassIncompleteError unless xml_path is complete OSM XML without an Overpass runtime error."""
    root_tag = None
    try:
        for event, elem in ET.iterparse(xml_path, events=("start", "end")):
            if event == "start":
                if root_tag is None:
                    root_tag = elem.tag
                    if root_tag != "osm":
                        raise OverpassIncompleteError(f"{xml_path} is not OSM XML (root tag: <{root_tag}>)")
                continue
            if elem.tag == "remark" and elem.text:
                remark = elem.text.strip()
                # Overpass returns HTTP 200 with partial data and this remark when it stops a query
                if "runtime error" in remark:
                    raise OverpassIncompleteError(f"Overpass stopped the query: {remark}")
                logger.warning("Overpass API remark: %s", remark)
            elif elem.tag in ("node", "way", "relation"):
                elem.clear()
    except ET.ParseError as e:
        raise OverpassIncompleteError(f"{xml_path} is truncated or not valid XML: {e}") from e


def download_osm_data_from_overpass_api(
    query_str: str, output_file_path: str, session: requests.Session, api_url: str
) -> str:
    # Download to a .part file so output_file_path only ever holds a complete download
    part_file_path = f"{output_file_path}.part"
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        with session.post(
            api_url,
            data=query_str,
            timeout=(30, get_read_timeout(query_str)),
            stream=True,
            headers={"User-Agent": USER_AGENT},
        ) as overpass_request:
            overpass_request.raise_for_status()
            total = int(overpass_request.headers.get("Content-Length", 0))
            task = progress.add_task("Requesting Overpass data ...", total=total if total > 0 else None)
            with open(part_file_path, "wb") as output_file:
                for chunk in overpass_request.iter_content(chunk_size=1024 * 128):
                    if not chunk:
                        continue
                    output_file.write(chunk)
                    if total > 0:
                        progress.update(task, advance=len(chunk))
    validate_osm_xml(part_file_path)
    os.replace(part_file_path, output_file_path)
    logger.info(f"Downloaded osm data to: {output_file_path}")
    return output_file_path


def download_with_retries(
    query_str: str,
    output_file_path: str,
    session: requests.Session,
    api_urls: list[str] | None = None,
    max_attempts: int = OVERPASS_MAX_ATTEMPTS,
    base_delay: float = OVERPASS_RETRY_BASE_DELAY,
) -> str:
    """Download query_str to output_file_path, backing off and rotating through api_urls on failure."""
    api_urls = api_urls or OVERPASS_API_URLS
    for attempt in range(1, max_attempts + 1):
        api_url = api_urls[(attempt - 1) % len(api_urls)]
        logger.info("Download attempt %d/%d from %s", attempt, max_attempts, api_url)
        try:
            return download_osm_data_from_overpass_api(query_str, output_file_path, session, api_url)
        except requests.exceptions.HTTPError as e:
            # A malformed query fails the same way every time
            if e.response is not None and e.response.status_code == 400:
                raise OverpassDownloadError(f"Overpass rejected the query: {e}") from e
            error = e
        except (requests.exceptions.RequestException, OverpassIncompleteError) as e:
            error = e
        logger.warning("Download attempt %d/%d failed: %s", attempt, max_attempts, error)
        if attempt < max_attempts:
            delay = min(base_delay * 2 ** (attempt - 1), OVERPASS_RETRY_MAX_DELAY) + random.uniform(0, base_delay)
            logger.info("Retrying in %.0f seconds", delay)
            time.sleep(delay)
    raise OverpassDownloadError(
        f"Failed to download {output_file_path} after {max_attempts} attempts: {error}"
    ) from error
