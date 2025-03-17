"""
Academic search engines for OpenManus.

This module provides specialized search engines for academic content,
including ArXiv, PubMed, and Semantic Scholar.
"""

import datetime
import json
import urllib.parse
import urllib.request
from typing import Any, Dict, Generator, List, Optional, Union

import requests
from loguru import logger

from app.tool.search.base import WebSearchEngine, SearchResult


class ArxivSearchEngine(WebSearchEngine):
    """
    Search engine for ArXiv papers.
    
    This engine searches the ArXiv preprint repository using their API.
    """
    
    name: str = "arxiv"
    
    def perform_search(
        self, 
        query: str, 
        num_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        date_range: Optional[Dict[str, str]] = None,
        categories: Optional[List[str]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Search ArXiv for papers matching the query.
        
        Args:
            query: Search query
            num_results: Max number of results to return
            sort_by: Sort by relevance, lastUpdatedDate, or submittedDate
            sort_order: Sort order (ascending or descending)
            date_range: Date range (e.g., {"start": "2020-01-01", "end": "2021-01-01"})
            categories: List of ArXiv categories (e.g., ["cs.AI", "cs.CL"])
            
        Returns:
            Generator of result URLs
        """
        try:
            # Build query string
            search_query = urllib.parse.quote(query)
            
            # Add categories if provided
            if categories:
                cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
                search_query = f"{search_query} AND ({cat_query})"
            
            # Add date range if provided
            if date_range and "start" in date_range:
                start_date = date_range["start"]
                search_query = f"{search_query} AND submittedDate:[{start_date}000000 TO *]"
                
                if "end" in date_range:
                    end_date = date_range["end"]
                    search_query = f"{search_query} AND submittedDate:[* TO {end_date}235959]"
            
            # Build API URL
            base_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": search_query,
                "start": 0,
                "max_results": num_results,
                "sortBy": "relevance" if sort_by == "relevance" else "lastUpdatedDate",
                "sortOrder": sort_order,
            }
            
            # Construct URL with parameters
            url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
            
            # Make request
            with urllib.request.urlopen(url) as response:
                import xml.etree.ElementTree as ET
                
                # Parse XML response
                tree = ET.parse(response)
                root = tree.getroot()
                
                # Extract entries
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    # Get URL
                    link = entry.find('{http://www.w3.org/2005/Atom}id')
                    if link is not None and link.text:
                        # Convert from arXiv API URL to web URL
                        paper_id = link.text.split('/')[-1]
                        paper_url = f"https://arxiv.org/abs/{paper_id}"
                        yield paper_url
        
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            # Return empty generator
            return
            yield from []


class PubMedSearchEngine(WebSearchEngine):
    """
    Search engine for PubMed medical publications.
    
    This engine searches the PubMed database using their E-utilities API.
    """
    
    name: str = "pubmed"
    
    def perform_search(
        self, 
        query: str, 
        num_results: int = 10,
        date_range: Optional[Dict[str, str]] = None,
        journal: Optional[str] = None,
        free_full_text: bool = False,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Search PubMed for medical publications matching the query.
        
        Args:
            query: Search query
            num_results: Max number of results to return
            date_range: Date range (e.g., {"start": "2020-01-01", "end": "2021-01-01"})
            journal: Filter by journal
            free_full_text: Only include free full text articles
            
        Returns:
            Generator of result URLs
        """
        try:
            # Build query
            search_query = query
            
            # Add journal filter
            if journal:
                search_query = f"{search_query} AND {journal}[Journal]"
            
            # Add date range
            if date_range and "start" in date_range:
                start_date = date_range["start"].replace("-", "/")
                if "end" in date_range:
                    end_date = date_range["end"].replace("-", "/")
                    search_query = f"{search_query} AND ({start_date}[Date - Publication] : {end_date}[Date - Publication])"
                else:
                    search_query = f"{search_query} AND {start_date}[Date - Publication]"
            
            # Add full text filter
            if free_full_text:
                search_query = f"{search_query} AND free full text[filter]"
            
            # URL encode the search query
            search_query = urllib.parse.quote(search_query)
            
            # First, search for IDs
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={search_query}&retmode=json&retmax={num_results}"
            
            # Make search request
            response = requests.get(search_url)
            if response.status_code != 200:
                logger.error(f"PubMed search error: {response.status_code}")
                return
                yield from []
            
            # Parse JSON response
            search_data = response.json()
            if 'esearchresult' not in search_data or 'idlist' not in search_data['esearchresult']:
                logger.error("Invalid PubMed search response")
                return
                yield from []
            
            # Get IDs
            id_list = search_data['esearchresult']['idlist']
            if not id_list:
                logger.info("No PubMed results found")
                return
                yield from []
            
            # Convert IDs to URLs
            for pmid in id_list:
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                yield pubmed_url
                
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            # Return empty generator
            return
            yield from []


class SemanticScholarSearchEngine(WebSearchEngine):
    """
    Search engine for Semantic Scholar academic papers.
    
    This engine searches Semantic Scholar using their API.
    """
    
    name: str = "semanticscholar"
    
    def perform_search(
        self, 
        query: str, 
        num_results: int = 10,
        year_range: Optional[Dict[str, int]] = None,
        fields_of_study: Optional[List[str]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Search Semantic Scholar for papers matching the query.
        
        Args:
            query: Search query
            num_results: Max number of results to return
            year_range: Year range (e.g., {"start": 2020, "end": 2021})
            fields_of_study: List of fields (e.g., ["Computer Science", "Medicine"])
            
        Returns:
            Generator of result URLs
        """
        try:
            # Build API URL
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": num_results,
                "fields": "url,year,fieldsOfStudy",
            }
            
            # Construct URL with parameters
            url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
            
            # Make request
            headers = {
                "Accept": "application/json",
                "User-Agent": "OpenManus-Agent/1.0",
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Semantic Scholar search error: {response.status_code}")
                return
                yield from []
            
            # Parse response
            data = response.json()
            if 'data' not in data:
                logger.error("Invalid Semantic Scholar response format")
                return
                yield from []
            
            # Filter results
            for paper in data['data']:
                # Apply year filter if provided
                if year_range and 'year' in paper:
                    year = paper['year']
                    if year is None:
                        continue
                    
                    if 'start' in year_range and year < year_range['start']:
                        continue
                    if 'end' in year_range and year > year_range['end']:
                        continue
                
                # Apply field filter if provided
                if fields_of_study and 'fieldsOfStudy' in paper and paper['fieldsOfStudy']:
                    if not any(field in paper['fieldsOfStudy'] for field in fields_of_study):
                        continue
                
                # Get URL
                if 'url' in paper and paper['url']:
                    yield paper['url']
        
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            # Return empty generator
            return
            yield from []


class GoogleScholarSearchEngine(WebSearchEngine):
    """
    Wrapper for Google search with Scholar-specific parameters.
    
    This is a wrapper around GoogleSearchEngine that adds Google Scholar specific parameters.
    Note: This uses regular Google search with site restriction since Google Scholar 
    doesn't have a public API.
    """
    
    name: str = "scholar"
    
    def __init__(self):
        from app.tool.search import GoogleSearchEngine
        self.google_engine = GoogleSearchEngine()
    
    def perform_search(
        self, 
        query: str, 
        num_results: int = 10,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Search Google Scholar for academic papers.
        
        Args:
            query: Search query
            num_results: Max number of results to return
            
        Returns:
            Generator of result URLs
        """
        # Modify query to search Google Scholar
        scholar_query = f"{query} site:scholar.google.com"
        
        # Use the Google search engine
        return self.google_engine.perform_search(scholar_query, num_results)
