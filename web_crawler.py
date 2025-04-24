# web_crawler.py
"""Handles fetching and parsing content from web URLs."""

import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse # For basic URL validation/normalization
from utils import log_message
from config import CRAWLING_TIMEOUT, MAX_CRAWL_TEXT_LENGTH

def is_valid_url(url):
    """Basic check if a string looks like a valid HTTP/HTTPS URL."""
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except ValueError:
        return False

def crawl_url(url):
    """
    Fetches and parses text content from a URL.
    Uses session state for caching results. Returns text or None on failure.
    """
    if "crawled_data" not in st.session_state:
        st.session_state.crawled_data = {}

    # Basic normalization (lowercase scheme/host, remove fragment)
    try:
        parsed = urlparse(url)
        normalized_url = parsed._replace(fragment="").geturl()
        if not is_valid_url(normalized_url):
             log_message(f"Skipping invalid or non-HTTP/S URL: {url}", "warning")
             return None
    except Exception as parse_err:
         log_message(f"Could not parse URL '{url}': {parse_err}", "warning")
         return None


    if normalized_url in st.session_state.crawled_data:
        log_message(f"Using cached content for URL: {normalized_url}", "debug")
        return st.session_state.crawled_data.get(normalized_url) # Return cached (could be None)

    log_message(f"Attempting to crawl URL: {normalized_url}")
    try:
        # Be a polite crawler
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; QABot/1.0; +http://example.com/bot)'}
        response = requests.get(normalized_url, timeout=CRAWLING_TIMEOUT, headers=headers, allow_redirects=True)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Check final URL after redirects
        final_url = response.url
        if final_url != normalized_url:
            log_message(f"Redirected to: {final_url}", "debug")
            # Optional: Cache result under final URL as well?

        content_type = response.headers.get('content-type', '').lower()
        crawled_text = None

        if 'html' in content_type:
            soup = BeautifulSoup(response.content, 'lxml')
            # Remove script, style, nav, footer, header, form elements etc.
            for element_type in ["script", "style", "nav", "footer", "aside", "header", "form", "button", "input", "meta", "link"]:
                 for element in soup.find_all(element_type):
                      element.decompose()
            # Get text, join paragraphs, limit length
            body = soup.find('body')
            if body:
                text_parts = [p.get_text(separator=' ', strip=True) for p in body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th'])]
                crawled_text = '\n'.join(filter(None, text_parts)) # Join non-empty parts
                crawled_text = crawled_text[:MAX_CRAWL_TEXT_LENGTH]
            else: # Fallback if no body tag found
                 crawled_text = ' '.join(soup.stripped_strings)[:MAX_CRAWL_TEXT_LENGTH]

            log_message(f"Successfully crawled and parsed HTML ({len(crawled_text)} chars) from {final_url}", "info")

        elif 'text/plain' in content_type:
             crawled_text = response.text[:MAX_CRAWL_TEXT_LENGTH]
             log_message(f"Successfully crawled plain text ({len(crawled_text)} chars) from {final_url}", "info")

        # Add more content types if needed (e.g., application/pdf - harder to parse)

        else:
            log_message(f"Skipping unsupported content type '{content_type}' at {final_url}", "warning")
            crawled_text = None

        # Cache the result (even if None) for the original normalized URL
        st.session_state.crawled_data[normalized_url] = crawled_text
        # Optional: Cache under the final URL too if different
        if final_url != normalized_url:
             st.session_state.crawled_data[final_url] = crawled_text

        return crawled_text

    except requests.exceptions.Timeout:
        log_message(f"Failed to crawl {normalized_url}: Timeout after {CRAWLING_TIMEOUT} seconds", "error")
        st.session_state.crawled_data[normalized_url] = None
        return None
    except requests.exceptions.RequestException as e:
        # Handle specific errors like connection errors, SSL errors, etc.
        log_message(f"Failed to crawl {normalized_url}: RequestException {e}", "error")
        st.session_state.crawled_data[normalized_url] = None
        return None
    except Exception as e:
        # Catch other potential errors during parsing etc.
        log_message(f"Error processing content from {normalized_url}: {e}", "error")
        st.session_state.crawled_data[normalized_url] = None
        return None
