from typing import Dict, List, Optional, Any

class SearchResult:
    """
    Represents a search result from any search engine.
    """
    def __init__(self, url: str, title: Optional[str] = None, snippet: Optional[str] = None):
        self.url = url
        self.title = title
        self.snippet = snippet
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet
        }

class WebSearchEngine(object):
    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> list[dict]:
        """
        Perform a web search and return a list of URLs.

        Args:
            query (str): The search query to submit to the search engine.
            num_results (int, optional): The number of search results to return. Default is 10.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            List: A list of dict matching the search query.
        """
        raise NotImplementedError
