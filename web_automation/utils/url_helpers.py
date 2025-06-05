# web_automation/utils/url_helpers.py
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def get_domain(url: str) -> str:
    """
    Extracts the domain (e.g., 'example.com') from a URL.
    """
    if not url:
        return ""
    try:
        return urlparse(url).netloc
    except Exception as e:
        logger.error(f"Error parsing URL '{url}' to get domain: {e}")
        return ""

def is_similar_url(url1: str, url2: str, ignore_scheme=True, ignore_query=True, ignore_fragment=True) -> bool:
    """
    Checks if two URLs are similar, potentially ignoring scheme, query params, and fragments.
    A more robust implementation would compare paths more carefully.
    """
    if not url1 or not url2:
        return False
    try:
        p1 = urlparse(url1)
        p2 = urlparse(url2)

        if ignore_scheme:
            scheme_match = True
        else:
            scheme_match = p1.scheme == p2.scheme
        
        netloc_match = p1.netloc == p2.netloc
        path_match = p1.path.strip('/') == p2.path.strip('/') # Basic path comparison

        if ignore_query:
            query_match = True
        else:
            query_match = p1.query == p2.query # Simple query comparison

        if ignore_fragment:
            fragment_match = True
        else:
            fragment_match = p1.fragment == p2.fragment
            
        return scheme_match and netloc_match and path_match and query_match and fragment_match
    except Exception as e:
        logger.error(f"Error comparing URLs '{url1}' and '{url2}': {e}")
        return False

if __name__ == '__main__':
    # Basic tests
    print(f"Domain of 'http://www.example.com/path?query=1#frag': {get_domain('http://www.example.com/path?query=1#frag')}")
    print(f"Domain of 'https://sub.example.co.uk': {get_domain('https://sub.example.co.uk')}")
    
    url_a = "http://example.com/test"
    url_b = "https://example.com/test?param=1"
    url_c = "http://example.com/Test/"
    url_d = "http://another.com/test"

    print(f"'{url_a}' vs '{url_b}' (ignore query): {is_similar_url(url_a, url_b)}") # True
    print(f"'{url_a}' vs '{url_b}' (strict query): {is_similar_url(url_a, url_b, ignore_query=False)}") # False
    print(f"'{url_a}' vs '{url_c}': {is_similar_url(url_a, url_c)}") # True (due to path.strip('/'))
    print(f"'{url_a}' vs '{url_d}': {is_similar_url(url_a, url_d)}") # False
