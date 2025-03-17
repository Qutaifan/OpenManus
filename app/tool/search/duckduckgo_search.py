from duckduckgo_search import DDGS

from app.tool.search.base import WebSearchEngine


class DuckDuckGoSearchEngine(WebSearchEngine):
    def perform_search(self, query, num_results=10, *args, **kwargs):
        """DuckDuckGo search engine.
        
        Note: This is a synchronous method that will be called in a thread by WebSearch tool.
        """
        try:
            # Convert num_results to int to ensure no type comparison issues
            num_results = int(num_results)
            
            # The duckduckgo_search library expects the parameter to be named 'max_results'
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=num_results)
                # Extract URLs from results
                return [result['href'] for result in results]
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            # Return empty list in case of error to allow fallback to other search engines
            return []
