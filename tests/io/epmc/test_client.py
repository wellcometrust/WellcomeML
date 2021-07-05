import pytest

from wellcomeml.io.epmc.client import EPMCClient

@pytest.fixture
def epmc_client():
    return EPMCClient(
        max_retries=3
    )

def test_search(epmc_client):
    pmid = '24889800'
    query = "ext_id:%s" % pmid
    session = epmc_client.requests_session()
    search_result = epmc_client.search(session, query)
    assert search_result['pmid']==pmid

def test_get_references(epmc_client):
    pub_id = '22749442'
    source = 'MED'
    session = epmc_client.requests_session()
    references = epmc_client.get_references(session, pub_id, source=source)
    assert len(references)!=0

def test_get_citations(epmc_client):
    pub_id = '22749442'
    source = 'MED'
    session = epmc_client.requests_session()
    citations = epmc_client.get_citations(session, pub_id, source=source)
    assert len(citations)!=0
    
