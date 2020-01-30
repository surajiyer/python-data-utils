# coding: utf-8

"""
        description: Scraper for extracting text from web URLs.
        author: Suraj Iyer
"""

__all__ = ['WebTextScraper']

from bs4 import BeautifulSoup
from bs4.element import Comment
import requests
from collections import OrderedDict
from typing import Iterable


class WebTextScraper:
    __cache = OrderedDict()
    __cache_limit: int = -1
    __exclude_tags = frozenset(('style', 'script', 'head', 'title', 'meta', '[document]'))
    exclude_tags = set()
    __user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36"

    def set_cache_limit(self, limit: int = -1):
        assert limit != 0, 'Limit cannot be zero.'
        self.__cache_limit = limit

        # delete urls from cache in order of insertion (FIFO)
        self.fix_cache_limit()

    def clear_cache(self):
        self.__cache = OrderedDict()

    def fix_cache_limit(self):
        if self.__cache_limit < 0:
            return
        while len(self.__cache) > self.__cache_limit:
            self.__cache.popitem(False)

    def drop_url_from_cache(self, url: str):
        if url in self.__cache:
            del self.__cache[url]

    def tag_visible(self, element) -> bool:
        if element.parent.name in self.__exclude_tags | self.exclude_tags:
            return False
        if isinstance(element, Comment):
            return False
        return True

    def text_from_html(self, body: str) -> str:
        soup = BeautifulSoup(body, 'html.parser')
        texts = soup.findAll(text=True)
        visible_texts = filter(self.tag_visible, texts)
        return u" ".join(t.strip() for t in visible_texts).strip()

    def text_from_url(
            self, url: str, timeout: int = 5, cache_it: bool = True,
            user_agent: str = None, ignore_errors: bool = True) -> str:
        if url in self.__cache:
            return self.text_from_html(self.__cache[url])

        page_response = requests.get(
            url, timeout=timeout,
            headers={
                'User-Agent': user_agent if user_agent else self.__user_agent})
        if 200 <= page_response.status_code.real < 300:
            text = self.text_from_html(page_response.content)

            if cache_it and url not in self.__cache:
                self.__cache[url] = page_response.content
                self.fix_cache_limit()
        else:
            print(url, 'Status: {}'.format(page_response.status_code.real))
            if ignore_errors:
                text = ""
            else:
                raise ValueError('{}. Got bad response. Code: {}'.format(
                    url, page_response.status_code.real))

        return text

    def urls_from_html(self, body: str) -> set:
        soup = BeautifulSoup(body, 'html.parser')
        urls = {a.get('href') for a in soup.find_all('a', href=True)}
        # Remove slash at the end
        urls = {url[:-1] if url[-1] ==
                '/' else url for url in urls if url[:-1] != ''}
        urls = {url for url in urls if url[0] != '#' or url.startswith(
            'javascript') or url.startswith('mailto')}  # Remove id references
        return urls

    def urls_from_url(
            self, url: str, timeout: int = 5, cache_it: bool = True,
            user_agent: str = None, ignore_errors: bool = True) -> set:
        if url in self.__cache:
            urls = self.urls_from_html(self.__cache[url])
            urls = {
                url + link if link.startswith('/') else link for link in urls}
            return urls

        page_response = requests.get(
            url, timeout=timeout,
            headers={
                'User-Agent': user_agent if user_agent else self.__user_agent})
        if 200 <= page_response.status_code.real < 300:
            urls = self.urls_from_html(page_response.content)
            urls = {
                url + link if link.startswith('/') else link for link in urls}

            if cache_it and url not in self.__cache:
                self.__cache[page_response.url[:-1]] = page_response.content
                self.fix_cache_limit()
        else:
            print(url, 'Status: {}'.format(page_response.status_code.real))
            if ignore_errors:
                urls = {}
            else:
                raise ValueError('{}. Got bad response. Code: {}'.format(
                    url, page_response.status_code.real))

        return urls

    def scrape_text(
            self, start_url: str, search_depth: int = 2,
            allowed_domains: Iterable[str] = [],
            user_agent: str = None, ignore_errors: bool = True) -> set:
        assert search_depth > -1, 'Search depth must be >= 0.'
        current_depth = 0
        to_visit = set(start_url)
        visited = set()

        while to_visit:
            for url in list(to_visit):
                yield self.text_from_url(
                    url, user_agent=user_agent, ignore_errors=ignore_errors)

                visited.add(url)
                to_visit.remove(url)

                if current_depth < search_depth:
                    urls = self.urls_from_url(
                        url, user_agent=user_agent,
                        ignore_errors=ignore_errors)
                    if not allowed_domains:
                        to_visit.add(
                            set(url for url in urls if url not in visited))
                    else:
                        to_visit.add(set(url for url in urls if url not in visited and any(
                            domain in url for domain in allowed_domains)))
            current_depth += 1

        assert len(to_visit) == 0, 'Shit, this should not have happend.'

        return visited