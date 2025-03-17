from googlesearch import search

from app.tool.search.base import WebSearchEngine


class GoogleSearchEngine(WebSearchEngine):
    def perform_search(self, query, num_results=10, *args, **kwargs):
        """Google search engine."""
        try:
            # Convert num_results to int to ensure no type comparison issues
            num_results = int(num_results)
            
            # Call the search function and return results
            results = search(query, num_results=num_results)
            # Ensure we return a list of URLs
            return list(results)
        except Exception as e:
            print(f"Google search error: {e}")
            # Return empty list in case of error to allow fallback to other search engines
            return []
