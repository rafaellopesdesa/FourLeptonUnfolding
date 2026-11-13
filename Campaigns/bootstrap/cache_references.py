"""Cache cited primary-source HTML for the study reading register, never Git."""
from html.parser import HTMLParser
from pathlib import Path
import concurrent.futures
import hashlib
import json
import urllib.request

ROOT = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/references')
PAPERS = ['1911.09107v2', '1208.4018v3', '2512.07074v1', '2412.01600v2',
          '2412.01548v2', '2606.11083v2', '2005.09375v1', '2109.07851v2',
          '2012.00578v2', '2308.13362v2', '1908.00005v2', '2309.05471v2',
          '2012.11343v3', '1405.0301v2']
class Text(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.skip += 1
        if tag in ('p', 'h1', 'h2', 'h3', 'h4', 'li', 'tr', 'section'): self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ('script', 'style'): self.skip = max(0, self.skip - 1)
        if tag in ('p', 'h1', 'h2', 'h3', 'h4', 'li', 'tr'): self.parts.append('\n')
    def handle_data(self, data):
        if not self.skip: self.parts.append(data)

def fetch(identifier):
    url = 'https://arxiv.org/html/' + identifier
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'FourLeptonStudy reference audit'})
        with urllib.request.urlopen(request, timeout=45) as response: data = response.read()
        (ROOT / (identifier + '.html')).write_bytes(data)
        parser = Text(); parser.feed(data.decode())
        lines = [' '.join(line.split()) for line in ''.join(parser.parts).splitlines()]
        text = '\n'.join(line for line in lines if line)
        (ROOT / (identifier + '.txt')).write_text(text)
        return {'id': identifier, 'url': url, 'status': 'cached', 'sha256': hashlib.sha256(data).hexdigest(),
                'characters': len(text), 'reading_status': 'not inferred from download'}
    except Exception as error:
        return {'id': identifier, 'url': url, 'status': 'failed', 'error': str(error)}
if __name__ == '__main__':
    ROOT.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch, PAPERS))
    (ROOT / 'cache_manifest.json').write_text(json.dumps(results, indent=2) + '\n')
    for result in results: print(result)
