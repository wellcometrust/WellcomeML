from unittest.mock import MagicMock
import pytest

from wellcomeml.io.epmc.client import EPMCClient


@pytest.fixture
def epmc_client():
    return EPMCClient(
        max_retries=3
    )


def test_search(epmc_client):
    epmc_client._execute_query = MagicMock()
    epmc_client.search(
        "session", "query", result_type="not core",
        page_size=15, only_first=False
    )

    expected_params = {
        "query": "query",
        "format": "json",
        "resultType": "not core",
        "pageSize": 15
    }
    epmc_client._execute_query.assert_called_with("session", expected_params, False)


def test_search_by_pmid(epmc_client):
    epmc_client.search = MagicMock(return_value="results")
    epmc_client.search_by_pmid("session", "pmid")
    epmc_client.search.assert_called_with("session", "ext_id:pmid", only_first=True)


def test_search_by_doi(epmc_client):
    epmc_client.search = MagicMock(return_value="results")
    epmc_client.search_by_doi("session", "doi")
    epmc_client.search.assert_called_with("session", "doi:doi", only_first=True)


def test_search_by_pmcid(epmc_client):
    epmc_client.search = MagicMock(return_value="results")
    epmc_client.search_by_pmcid("session", "PMCID0")
    epmc_client.search.assert_called_with("session", "pmcid:PMCID0", only_first=True)


def test_search_by_invalid_pmcid(epmc_client):
    epmc_client.search = MagicMock(return_value="results")
    with pytest.raises(ValueError):
        epmc_client.search_by_pmcid("session", "pmcid")


def test_get_full_text(epmc_client):
    epmc_client._get_response_content = MagicMock(return_value="content")
    epmc_client.get_full_text("session", "pmid")

    epmc_endpoint = epmc_client.api_endpoint
    epmc_client._get_response_content.assert_called_with(
        "session",
        f"{epmc_endpoint}/pmid/fullTextXML"
    )


def test_get_references(epmc_client):
    epmc_client._get_response_json = MagicMock(return_value={"references": []})
    epmc_client.get_references("session", "pmid")

    epmc_endpoint = epmc_client.api_endpoint
    params = {"format": "json", "page": 1, "pageSize": 1000}
    epmc_client._get_response_json.assert_called_with(
        "session",
        f"{epmc_endpoint}/MED/pmid/references",
        params
    )


def test_get_citations(epmc_client):
    epmc_client._get_response_json = MagicMock(return_value={"references": []})
    epmc_client.get_citations("session", "pmid")

    epmc_endpoint = epmc_client.api_endpoint
    params = {"format": "json", "page": 1, "pageSize": 1000}
    epmc_client._get_response_json.assert_called_with(
        "session",
        f"{epmc_endpoint}/MED/pmid/citations",
        params
    )
