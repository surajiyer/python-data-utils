# coding: utf-8

"""
    description: Scraper for extracting text from web URLs.
    author: Suraj Iyer
"""

__all__ = ['WebTextScraper', '_URL', '_HTML']

from bs4 import BeautifulSoup
from bs4.element import Comment
from python_data_utils.nlp.web_scraper.requests_html import HTMLSession
from collections import OrderedDict
from typing import Any, Awaitable, Dict, Callable, Iterable,\
    Generator, Set, Tuple, Union
import random
from requests.exceptions import ReadTimeout, ConnectTimeout
from pyppeteer.errors import NetworkError
import asyncio
import traceback


# Typing
_URL = str
_HTML = str


class WebTextScraper:
    __cache: Dict[_URL, str] = OrderedDict()
    __cache_limit: int = -1
    __default_exclude_tags = frozenset(('style', 'script', 'head', 'title', 'meta', '[document]'))
    exclude_tags: Set[str] = set()
    exclude_class: Set[Union[str, Callable[[Any], bool]]] = set()
    __default_user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36"
    session = HTMLSession()
    proxies_list: Iterable[Dict[str, str]] = []

    def __init__(self, max_redirects: int = 30, path_to_proxies: str = None):
        self.session.max_redirects = max_redirects
        if path_to_proxies:
            self.load_proxies(path_to_proxies)

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

    def drop_url_from_cache(self, url: _URL):
        if url in self.__cache:
            del self.__cache[url]

    def load_proxies(self, path):
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                self.proxies_list.append({'http': line,'https': line})
                line = f.readline()

    def element_visible(self, element) -> bool:
        return not (
            len(element.strip()) == 0
            or element.parent.name in self.__default_exclude_tags | self.exclude_tags
            or isinstance(element, Comment)
            or any(element.find_parents(attrs={"class": c}) for c in self.exclude_class))

    def text_from_html(self, body: _HTML) -> str:
        soup = BeautifulSoup(body, 'html.parser')
        texts = soup.findAll(text=True)
        visible_texts = filter(self.element_visible, texts)
        return u" ".join(t.strip() for t in visible_texts)\
            .replace('\n', ' ')\
            .replace('\r', ' ')\
            .strip()

    def urls_from_html(self, body: _HTML) -> Set[_URL]:
        soup = BeautifulSoup(body, 'html.parser')
        urls = {a.get('href') for a in soup.find_all('a', href=True)}
        # Remove slash at the end
        urls = {url[:-1] if url[-1] ==
                '/' else url for url in urls if url[:-1] != ''}
        urls = {url for url in urls if url[0] != '#' or url.startswith(
            'javascript') or url.startswith('mailto')}  # Remove id references
        return urls

    def html_body_from_url(
            self, *, url: _URL, timeout: int = 5, cache_it: bool = True,
            user_agent: str = None, ignore_response_errors: bool = False,
            js_render: bool = False, js_render_kws: dict = {},
            wait_threshold: Union[float, int] = 5,
            allow_fragments: bool = False,
            js_render_page_cb: Callable[['pyppeteer.page.Page'], Awaitable[_HTML]] = None) -> _HTML:
        if not allow_fragments:
            url = url.split('#')[0].split('?')[0]

        # serve from cache if available
        if url in self.__cache:
            return self.__cache[url]['html']

        try:
            r = self.session.get(
                url, timeout=timeout,
                proxies=random.choice(self.proxies_list) if self.proxies_list else None,
                headers={
                    'User-Agent': user_agent if user_agent else self.__default_user_agent})
        except (ReadTimeout, ConnectTimeout):
            if ignore_response_errors:
                result = ""
                if cache_it and url not in self.__cache:
                    self.__cache[url] = {'html': result}
            else:
                raise

        if 'result' not in locals():
            if 200 <= r.status_code.real < 300:
                # avoid Document is Empty errors
                if not r.content:
                    content = lambda: None
                    content.html = ""
                    js_render = False
                else:
                    content = r.html

                if js_render:
                    js_render_kws = js_render_kws.copy()

                    # Update timeout
                    js_render_kws.update({'timeout': js_render_kws.get('timeout', timeout)})

                    # Apply waiting before rendering to prevent timeouts
                    original_wait = js_render_kws.get('wait', 0.2)
                    add_wait = 0
                    rendered = False
                    while not rendered:
                        try:
                            js_render_kws.update({'wait': original_wait + add_wait})
                            script_result = content.render(**js_render_kws)
                            rendered = True
                        except NetworkError as e:
                            if traceback.format_exception_only(type(e), e)[0].strip()\
                                == "pyppeteer.errors.NetworkError: Execution context was destroyed, most likely because of a navigation.":
                                add_wait += 2
                                if original_wait + add_wait > wait_threshold:
                                    raise
                            else:
                                raise
                        except Exception as e:
                            raise

                    # if keep_page=True while rendering
                    if hasattr(content, 'page') and content.page:
                        # if callback function given for page object after rendering
                        if js_render_page_cb:
                            actual_url = content.page.url
                            (content.html,) = self.session.loop.run_until_complete(
                                asyncio.gather(js_render_page_cb(content.page, script_result)))

                        # close the page
                        self.session.loop.run_until_complete(
                            asyncio.gather(content.page.close()))
                        content.page = None

                result = content.html

                if cache_it and url not in self.__cache:
                    self.__cache[url] = {'html': result}
                    if 'actual_url' in locals() and url != actual_url:
                        self.__cache[actual_url] = {'html': result}
                    self.fix_cache_limit()
            else:
                if ignore_response_errors:
                    result = ""
                    if cache_it and url not in self.__cache:
                        self.__cache[url] = {'html': result}
                else:
                    raise ValueError('{}. Got bad response. Code: {}'.format(
                        url, r.status_code.real))

        return result

    def text_from_url(self, *args, **kwargs) -> str:
        return self.text_from_html(self.html_body_from_url(*args, **kwargs))

    def urls_from_url(self, *args, url: _URL, **kwargs) -> Set[_URL]:
        urls = self.urls_from_html(self.html_body_from_url(url=url, **kwargs))
        urls = {
            url + link if link.startswith('/') else link for link in urls}
        return urls

    def start_scraper_from_session(
            self, to_visit: Set[_URL], visited: Set[_URL],
            current_depth: int, search_depth: int,
            allowed_domains: Iterable[_URL] = [],
            allow_fragments: bool = False,
            html_body_from_urls_kws: dict = {}):
        assert current_depth >= 0, 'Current depth must be >= 0.'
        assert search_depth >= 0, 'Search depth must be >= 0.'
        html_body_from_urls_kws = html_body_from_urls_kws.copy()
        html_body_from_urls_kws.update({
            'allow_fragments': html_body_from_urls_kws.get('allow_fragments', allow_fragments)})

        if not allow_fragments:
            to_visit = set(url.split('#')[0].split('?')[0] for url in to_visit)
            visited = set(url.split('#')[0].split('?')[0] for url in visited)

        if allowed_domains:
            to_visit = set(url for url in to_visit if url not in visited and any(
                domain in url for domain in allowed_domains))
        else:
            to_visit = set(url for url in to_visit if url not in visited)

        while to_visit:
            for url in list(to_visit):
                try:
                    text = None
                    text = self.text_from_url(url=url, **html_body_from_urls_kws)
                    yield url, text

                    visited.add(url)
                    to_visit.remove(url)

                    if current_depth < search_depth:
                        urls = self.urls_from_url(url=url)

                        if not allow_fragments:
                            urls = set(url.split('#')[0].split('?')[0] for url in urls)

                        if not allowed_domains:
                            to_visit |= set(url for url in urls if url not in visited)
                        else:
                            to_visit |= set(url for url in urls if url not in visited and any(
                                domain in url for domain in allowed_domains))
                except:
                    traceback.print_exc()
                    self.scraper_session = {
                        'to_visit': to_visit,
                        'visited': visited,
                        'current_depth': current_depth,
                        'search_depth': search_depth,
                        'allowed_domains': allowed_domains,
                        'allow_fragments': allow_fragments,
                        'html_body_from_urls_kws': html_body_from_urls_kws}
                    yield url, None
                    return

            current_depth += 1

        assert len(to_visit) == 0, 'Shit, this should not have happened.'

    def start_scraper(
            self, start_url: _URL, search_depth: int = 2,
            allowed_domains: Iterable[_URL] = [],
            allow_fragments: bool = False,
            html_body_from_urls_kws: dict = {}) -> Generator[Tuple[_URL, str], None, None]:
        current_depth: int = 0
        to_visit = set({start_url})
        visited = set()
        return self.start_scraper_from_session(
            to_visit, visited, current_depth, search_depth
            , allowed_domains, allow_fragments, html_body_from_urls_kws)

    def close(self):
        self.session.close()
