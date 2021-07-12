"""
Client library for the EPMC REST API.
"""
import logging

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

RETRY_PARAMETERS = {
    'backoff_factor': 1.2,
    'status_forcelist': [429, 500, 502, 503, 504],
}

RETRY_BACKOFF_MAX = 310


class EPMCClient:

    EPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"

    def __init__(self, api_endpoint=EPMC_URL, max_retries=60, log_level=logging.INFO):
        self.api_endpoint = api_endpoint
        self.max_retries = max_retries

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

    def requests_session(self, adapter_kwargs=None):
        """
        Returns a requests session with retry params in place, and
        configured for a single persistent connection (since we
        generally expect one session per thread).
        """
        session = requests.Session()
        retries = Retry(**RETRY_PARAMETERS)
        retries.BACKOFF_MAX = RETRY_BACKOFF_MAX
        if adapter_kwargs is None:
            adapter_kwargs = {}
        session.mount('https://', HTTPAdapter(
            max_retries=retries, **adapter_kwargs)
                      )
        return session

    def _execute_query(self, session, params, only_first=True):
        """
        Try to execute the query for the given amount of retries, adding some
        sleep time every retry.
        Args:
            session: A requests session
            params: A string containing the query
            only_first: Whether to only return the first result (e.g. in
            the case of searching for a specific PMID or DOI). Will be
            deprecated
        Returns:
            The response of the query, as a dictionary
        """
        if only_first:
            self.logger.warning("Only returning first entry of search as a "
                                "dictionary. Will be deprecated in the future"
                                " and a list of arguments will be"
                                " returned by default.")

        epmc_search_url = "/".join([self.api_endpoint, "search"])

        go_to_next_page = True
        page = 1

        final_results = []

        while go_to_next_page:
            self.logger.debug(f"Querying results from "
                              f"{page*params['pageSize']} to "
                              f"{(page+1)*params['pageSize']}")

            response = session.get(
                epmc_search_url,
                params=params
            )

            response_json = response.json()

            current_page_result = response_json["resultList"]["result"]

            if not current_page_result or only_first:
                go_to_next_page = False
            else:
                params['cursorMark'] = response_json.get("nextCursorMark")

            final_results += current_page_result
            page += 1

        if final_results:
            return final_results[0] if only_first else final_results
        else:
            self.logger.warning(
                "epmc.license.get_pm_metadata: no-results query=%s", params
            )
            return None

    def search(self, session, query, result_type='core', page_size=10,
               only_first=True, **kwargs):
        """ Fetches metadata from EPMC's REST API for a query. """
        # Sample URL for searching by DOI:
        #   https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=doi:10.1111/hiv.12674

        params = {
            "query": query,
            "format": "json",
            "resultType": result_type,
            "pageSize": page_size
        }

        return self._execute_query(session, params, only_first, **kwargs)

    def search_by_doi(self, session, doi, **kwargs):
        query = "doi:%s" % doi
        return self.search(session, query, only_first=True, **kwargs)

    def search_by_pmid(self, session, pmid, **kwargs):
        query = "ext_id:%s" % pmid
        return self.search(session, query, only_first=True, **kwargs)

    def search_by_pmids(self, session, pmids, **kwargs):
        query = "ext_id:%s" % '"' + '" OR ext_id:"'.join(pmids) + '"'
        return self.search(session, query, only_first=False, **kwargs)

    def search_by_pmcid(self, session, pmcid, **kwargs):
        if not pmcid.startswith('PMC'):
            raise ValueError("Invalid PMCID: %s" % pmcid)
        query = "pmcid:%s" % pmcid
        return self.search(session, query, only_first=True, **kwargs)

    def search_by_pmcids(self, session, pmcids, **kwargs):
        query = "pmcid:%s" % '"' + '" OR pmcid:"'.join(pmcids) + '"'
        return self.search(session, query, only_first=False, **kwargs)

    def search_by_dois(self, session, dois, **kwargs):
        """
        Args:
            session: requests.Session
            dois: list of DOI strings

        Yields:
            Dicts of EPMC responses
        """
        query = '"' + '" OR "'.join(dois) + '"'
        return self.search(session=session, query=query, only_first=False, **kwargs)

    def _get_response_content(self, session, epmc_fulltext_url):
        response = session.get(epmc_fulltext_url)
        response.raise_for_status()
        # NB: requests will not reliably guess the encoding, and
        # we're passing the response directly into LXML, which will
        # happily parse bytes as well as str. Thus use response.content
        # (bytes), not response.text (str).
        return response.content

    def get_full_text(self, session, pmid):
        """ Fetches full text from EPMC's REST API for a given pmid. """
        try:
            epmc_fulltext_url = "/".join(
                [self.api_endpoint, str(pmid), "fullTextXML"]
            )
            return self._get_response_content(session, epmc_fulltext_url)
        except requests.RequestException as e:
            if hasattr(e, 'response'):
                if e.response.status_code == 404:
                    self.logger.info(
                        "epmc.get_full_text: http-404 pmid=%s", pmid
                    )
                    return None
                else:
                    self.logger.error(
                        "epmc.get_full_text: http-%d pmid=%s",
                        e.response.status_code,
                        pmid
                    )
            raise
        return None

    def _get_response_json(self, session, epmc_references_url, params):
        response = session.get(
            epmc_references_url,
            params=params
        )
        response_json = response.json()
        return response_json

    def get_references(self, session, pub_id, source='MED'):
        """
        Fetches references from EPMC's REST API for a given publication pub_id and source
        (e.g. 'MED')
        """
        params = {"format": "json", "page": 1, "pageSize": 1000}

        epmc_references_url = "/".join(
            [self.api_endpoint, source, str(pub_id), 'references']
        )

        go_to_next_page = True
        references = []

        while go_to_next_page:
            try:
                response_json = self._get_response_json(session, epmc_references_url, params)
                current_page_result = response_json.get('referenceList', {}).get('reference', [])
                if not current_page_result:
                    go_to_next_page = False
                else:
                    references += current_page_result
                    params['page'] += 1
            except requests.RequestException:
                self.logger.error(
                    "epmc.get_references: requests error pub_id=%s query=%s", pub_id, params
                )
                go_to_next_page = False

        if references:
            return references
        else:
            return None

    def get_citations(self, session, pub_id, source='MED'):
        """
        Fetches citations from EPMC's REST API for a given publication pub_id and source
        (e.g. 'MED')
        """

        params = {"format": "json", "page": 1, "pageSize": 1000}

        epmc_references_url = "/".join(
            [self.api_endpoint, source, str(pub_id), 'citations']
        )

        go_to_next_page = True
        citations = []

        while go_to_next_page:
            try:
                response_json = self._get_response_json(session, epmc_references_url, params)
                current_page_result = response_json.get('citationList', {}).get('citation', [])
                if not current_page_result:
                    go_to_next_page = False
                else:
                    citations += current_page_result
                    params['page'] += 1
            except requests.RequestException:
                self.logger.error(
                    "epmc.get_citations: requests error pub_id=%s query=%s", pub_id, params
                )
                go_to_next_page = False

        if citations:
            return citations
        else:
            return None
