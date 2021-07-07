from wellcomeml.io.epmc.client import EPMCClient

client = EPMCClient(max_retries=3)
session = client.requests_session()
pmid = "34215990"

references = client.get_references(session, pmid)
print(f"Found {len(references)} references")

result = client.search_by_pmid(session, pmid)
print(f"Found pub with keys {result.keys()}")
