# coding: utf-8

"""
    description: Spell check with Google Search's "Did you mean:" recommendation
    author: Suraj Iyer
    Original author: noahcoad@github
    URL: http://github.com/noahcoad/google-spell-check.
"""

import urllib
import re
import html


class GoogleSpellCheck:
    def correct(self, text):
        # grab html
        page = self.get_page('http://www.google.com/search?hl=en&q=' + urllib.parse.quote(text) + "&meta=&gws_rd=ssl")

        # save html for debugging
        # open('page.html', 'w').write(page)

        # pull pieces out
        match = re.search(r'(?:Showing results for|Did you mean|Including results for)[^\0]*?<a.*?>(.*?)</a>', page)
        if match is None:
            fix = text
        else:
            fix = match.group(1)
            fix = re.sub(r'<.*?>', '', fix)
            fix = html.unescape(fix)

        # return result
        return fix

    def get_page(self, url):
        # the type of header affects the type of response google returns
        # for example, using the commented out header below google does not
        # include "Including results for" results and gives back a different set of results
        # than using the updated user_agent yanked from chrome's headers
        # user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.116 Safari/537.36'
        headers = {'User-Agent': user_agent}
        req = urllib.request.Request(url, None, headers)
        page = urllib.request.urlopen(req)
        html = str(page.read())
        page.close()
        return html
