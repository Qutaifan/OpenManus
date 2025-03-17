from baidusearch.baidusearch import search

from app.tool.search.base import WebSearchEngine


class BaiduSearchEngine(WebSearchEngine):
    def perform_search(self, query, num_results=10, *args, **kwargs):
        """Baidu search engine."""
        try:
            # Convert num_results to int to ensure no type comparison issues
            num_results = int(num_results)
            
            # Call the search function and return results
            results = search(query, num_results=num_results)
            # Ensure we return a list of URLs
            return list(results)
        except Exception as e:
            print(f"Baidu search error: {e}")
            # Return empty list in case of error to allow fallback to other search engines
            return []
