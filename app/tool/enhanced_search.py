"""
Enhanced web search tools for OpenManus.

This module provides advanced search tools that combine multiple search engines,
including general web search and specialized academic search engines.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from loguru import logger
from pydantic import Field

from app.config import config
from app.recovery.recovery_manager import recoverable
from app.tool.base import BaseTool
from app.tool.search import (
    BaiduSearchEngine,
    DuckDuckGoSearchEngine,
    GoogleSearchEngine,
    WebSearchEngine,
)
from app.tool.search.academic_search import (
    ArxivSearchEngine,
    GoogleScholarSearchEngine,
    PubMedSearchEngine,
    SemanticScholarSearchEngine,
)
from app.tool.web_search import WebSearch


class AcademicSearch(BaseTool):
    """
    Academic search tool that retrieves papers and publications from academic sources.
    
    This tool searches across multiple academic search engines including
    ArXiv, PubMed, Semantic Scholar, and Google Scholar.
    """
    
    name: str = "academic_search"
    description: str = """
        Search for academic papers, research publications, and scholarly content.
        Results come from multiple academic databases including ArXiv, PubMed,
        Semantic Scholar, and Google Scholar.
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query for academic content.",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 10.",
                "default": 10,
            },
            "search_type": {
                "type": "string",
                "description": "(optional) Type of academic search to perform: 'general', 'papers', 'medicine', or 'computer_science'. Default is 'general'.",
                "enum": ["general", "papers", "medicine", "computer_science"],
                "default": "general",
            },
            "year_range": {
                "type": "object",
                "description": "(optional) Year range to filter results by.",
                "properties": {
                    "start": {
                        "type": "integer",
                        "description": "Start year (inclusive)."
                    },
                    "end": {
                        "type": "integer",
                        "description": "End year (inclusive)."
                    }
                }
            },
        },
        "required": ["query"],
    }
    
    # Search engines
    _search_engines: Dict[str, WebSearchEngine] = {
        "arxiv": ArxivSearchEngine(),
        "pubmed": PubMedSearchEngine(),
        "semantic_scholar": SemanticScholarSearchEngine(),
        "scholar": GoogleScholarSearchEngine(),
    }
    
    # Search engine configurations for different search types
    _search_configs: Dict[str, List[Tuple[str, int]]] = {
        "general": [
            ("scholar", 5),
            ("semantic_scholar", 5),
            ("arxiv", 3),
            ("pubmed", 2),
        ],
        "papers": [
            ("arxiv", 7),
            ("semantic_scholar", 5),
            ("scholar", 3),
        ],
        "medicine": [
            ("pubmed", 10),
            ("semantic_scholar", 5),
        ],
        "computer_science": [
            ("arxiv", 7),
            ("semantic_scholar", 5),
            ("scholar", 3),
        ],
    }
    
    @recoverable(max_retries=2)
    async def execute(
        self, 
        query: str, 
        num_results: int = 10, 
        search_type: str = "general",
        year_range: Optional[Dict[str, int]] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Execute academic search across multiple engines.
        
        Args:
            query: The search query for academic content
            num_results: Number of results to return
            search_type: Type of academic search to perform
            year_range: Year range for filtering results
            
        Returns:
            List of result dictionaries with source and URL
        """
        # Get search config based on search type
        search_config = self._search_configs.get(search_type, self._search_configs["general"])
        
        # Calculate results per engine
        results_per_engine = {}
        total_weight = sum(weight for _, weight in search_config)
        remaining = num_results
        
        for engine, weight in search_config:
            # Calculate share of results based on weight
            engine_results = max(1, int(round(num_results * weight / total_weight)))
            if engine_results > remaining:
                engine_results = remaining
            
            results_per_engine[engine] = engine_results
            remaining -= engine_results
        
        # Redistribute any remaining results to the first engine
        if remaining > 0 and search_config:
            first_engine = search_config[0][0]
            results_per_engine[first_engine] += remaining
        
        # Create tasks for each search engine
        tasks = []
        for engine_name, count in results_per_engine.items():
            if engine_name in self._search_engines:
                engine = self._search_engines[engine_name]
                
                # Create additional search params
                search_params = {"num_results": count}
                
                # Add year range if provided
                if year_range and engine_name in ["semantic_scholar", "arxiv"]:
                    if engine_name == "arxiv" and "start" in year_range:
                        # Convert to date format for ArXiv
                        start_year = year_range["start"]
                        end_year = year_range.get("end", start_year + 10)
                        
                        date_range = {
                            "start": f"{start_year}-01-01",
                            "end": f"{end_year}-12-31" if "end" in year_range else None
                        }
                        search_params["date_range"] = date_range
                    else:
                        search_params["year_range"] = year_range
                
                # Create specialized params based on engine and search type
                if engine_name == "arxiv" and search_type == "computer_science":
                    search_params["categories"] = ["cs.AI", "cs.CL", "cs.CV", "cs.LG"]
                
                # Create the search task
                task = self._search_with_engine(
                    engine_name, 
                    engine, 
                    query, 
                    search_params
                )
                tasks.append(task)
        
        # Execute all searches in parallel
        all_results = []
        if tasks:
            result_lists = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for results in result_lists:
                if isinstance(results, Exception):
                    logger.error(f"Search error: {results}")
                    continue
                
                if isinstance(results, list):
                    all_results.extend(results)
        
        # Deduplicate results by URL
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            url = result.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        # Limit to requested number
        return unique_results[:num_results]
    
    async def _search_with_engine(
        self, 
        engine_name: str,
        engine: WebSearchEngine,
        query: str,
        params: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Search with a specific engine and format results.
        
        Args:
            engine_name: Name of the search engine
            engine: The search engine to use
            query: Search query
            params: Search parameters
            
        Returns:
            List of search results as dictionaries
        """
        try:
            # Extract num_results from params
            num_results = params.pop("num_results", 10)
            
            # Run search in executor
            loop = asyncio.get_event_loop()
            urls = await loop.run_in_executor(
                None, 
                lambda: list(engine.perform_search(query, num_results=num_results, **params))
            )
            
            # Format results
            results = []
            for url in urls:
                results.append({
                    "source": engine_name,
                    "url": url,
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching with {engine_name}: {e}")
            return []


class EnhancedWebSearch(BaseTool):
    """
    Unified search tool that combines web search and academic search.
    
    This tool intelligently searches across both web and academic sources
    to provide comprehensive results for any query.
    """
    
    name: str = "enhanced_search"
    description: str = """
        Enhanced search that combines web and academic sources for comprehensive results.
        Intelligently distributes search across general web search engines (Google, DuckDuckGo, Baidu)
        and academic sources (ArXiv, PubMed, Semantic Scholar, Google Scholar) based on the query type.
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query.",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 10.",
                "default": 10,
            },
            "include_academic": {
                "type": "boolean",
                "description": "(optional) Whether to include academic sources. Default is true for queries that appear academic.",
                "default": None,
            },
            "search_mode": {
                "type": "string",
                "description": "(optional) Search mode to use: 'auto', 'web_only', 'academic_only', or 'combined'. Default is 'auto'.",
                "enum": ["auto", "web_only", "academic_only", "combined"],
                "default": "auto",
            },
        },
        "required": ["query"],
    }
    
    # Academic keywords to detect academic queries
    _academic_keywords: List[str] = [
        "research", "paper", "journal", "study", "publication", 
        "article", "conference", "thesis", "dissertation", "author", 
        "published", "doi", "arxiv", "ieee", "acm", "pubmed", "science",
        "scientific", "medicine", "medical", "biology", "chemistry", "physics",
        "mathematics", "computer science", "ai", "machine learning"
    ]
    
    # Regular expressions for detecting academic queries
    _academic_patterns: List[re.Pattern] = [
        re.compile(r"\b(paper|article|research|study)\s+(on|about)\b", re.IGNORECASE),
        re.compile(r"\b(find|search for)\s+(papers|articles|research|studies)\b", re.IGNORECASE),
        re.compile(r"\b(published|written)\s+(by|in)\b", re.IGNORECASE),
        re.compile(r"\b(journal|conference)\s+(of|on|in)\b", re.IGNORECASE),
        re.compile(r"\b(papers|articles)\s+(about|on|regarding)\b", re.IGNORECASE),
        re.compile(r"\b(doi|arxiv|isbn|issn)[\s:]\d", re.IGNORECASE),
    ]
    
    def __init__(self, **data):
        super().__init__(**data)
        # Initialize search tools
        self.web_search = WebSearch()
        self.academic_search = AcademicSearch()
    
    def _is_academic_query(self, query: str) -> bool:
        """
        Determine if a query is academic in nature.
        
        Args:
            query: The search query
            
        Returns:
            True if the query appears academic, False otherwise
        """
        # Convert to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Check for academic keywords
        for keyword in self._academic_keywords:
            if keyword.lower() in query_lower:
                return True
        
        # Check for academic patterns
        for pattern in self._academic_patterns:
            if pattern.search(query):
                return True
        
        return False
    
    @recoverable(max_retries=2)
    async def execute(
        self, 
        query: str, 
        num_results: int = 10,
        include_academic: Optional[bool] = None,
        search_mode: str = "auto",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute enhanced search across web and academic sources.
        
        Args:
            query: The search query
            num_results: Number of results to return
            include_academic: Whether to include academic sources
            search_mode: Search mode to use
            
        Returns:
            List of search results with source, URL, and title (if available)
        """
        # Determine if academic search should be included
        is_academic = self._is_academic_query(query) if include_academic is None else include_academic
        
        # Determine search mode
        if search_mode == "auto":
            if is_academic:
                search_mode = "combined"
            else:
                search_mode = "web_only"
        
        # Initialize results
        web_results = []
        academic_results = []
        
        # Execute searches based on mode
        if search_mode in ["web_only", "combined"]:
            # Determine how many web results to fetch
            web_count = num_results if search_mode == "web_only" else max(5, num_results // 2)
            web_results = await self.web_search.execute(query=query, num_results=web_count)
            
            # Convert to proper format
            web_results = [{"source": "web", "url": url} for url in web_results]
        
        if search_mode in ["academic_only", "combined"]:
            # Determine how many academic results to fetch
            academic_count = num_results if search_mode == "academic_only" else max(5, num_results // 2)
            academic_results = await self.academic_search.execute(
                query=query, 
                num_results=academic_count
            )
        
        # Combine results
        if search_mode == "combined":
            # Calculate total results available
            total_available = len(web_results) + len(academic_results)
            
            # If we don't have enough results, try to get more from whichever source has any
            if total_available < num_results:
                if len(web_results) > 0:
                    more_web = await self.web_search.execute(
                        query=query, 
                        num_results=num_results - total_available
                    )
                    web_results.extend([{"source": "web", "url": url} for url in more_web])
                elif len(academic_results) > 0:
                    more_academic = await self.academic_search.execute(
                        query=query, 
                        num_results=num_results - total_available
                    )
                    academic_results.extend(more_academic)
            
            # Interleave results, starting with more relevant source based on query type
            all_results = []
            if is_academic:
                # Start with academic results for academic queries
                for i in range(max(len(web_results), len(academic_results))):
                    if i < len(academic_results):
                        all_results.append(academic_results[i])
                    if i < len(web_results):
                        all_results.append(web_results[i])
            else:
                # Start with web results for non-academic queries
                for i in range(max(len(web_results), len(academic_results))):
                    if i < len(web_results):
                        all_results.append(web_results[i])
                    if i < len(academic_results):
                        all_results.append(academic_results[i])
                        
            results = all_results
        else:
            # Just use whichever results we fetched
            results = web_results + academic_results
        
        # Deduplicate results by URL
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        # Limit to requested number
        return unique_results[:num_results]
