import requests
from bs4 import BeautifulSoup as bs
from urllib.request import Request, urlopen
import os
import zipfile
import shutil
from sgf_to_position import process_selfplay


_URL = "https://katagoarchive.org/g170/selfplay/index.html"
PROJECT_ROOT = os.getcwd()
selfplay_dir = os.path.join(PROJECT_ROOT, "kata-selfplay")
refined_log_dir = os.path.join(PROJECT_ROOT, "refined_logs")
log_dir = os.path.join(PROJECT_ROOT, "analysis_logs")

# functional
r = requests.get(_URL)
soup = bs(r.text)
urls = []
names = []
_URL_PARENT = _URL.rsplit('/', 1)[0]
for i, link in enumerate(soup.findAll('a')):
    ZIP_NAME = link.get('href')[2:]  # The first characters are a dot and a slash
    _FULLURL = _URL_PARENT + '/' + ZIP_NAME
    # print(_FULLURL)
    if ZIP_NAME.endswith('.zip') and ZIP_NAME[0] == 'b' and ZIP_NAME[1] != '6' and ZIP_NAME[1:3] != '10' and not ZIP_NAME[:-4] in os.listdir(refined_log_dir):
        print('Downloading zip')
        rq = Request(url=_FULLURL,
                     headers={'User-Agent': 'Mozilla/5.0'})
        zip_opened = urlopen(rq)
        # res = urllib.urlopen(rq)
        # urls.append(_FULLURL)
        # names.append(soup.select('a')[i].attrs['href'])
        zip_path = os.path.join(selfplay_dir, ZIP_NAME)
        with open(zip_path, 'wb') as f:
            f.write(zip_opened.read())
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(selfplay_dir)
        # os.unlink(zip_path)
        process_selfplay()
        assert os.path.exists(os.path.join(refined_log_dir, ZIP_NAME[:-4]))
        shutil.rmtree(zip_path[:-4], ignore_errors=True)
        print('HA!')
        # break

# names_urls = zip(names, urls)

# for name, url in names_urls:
#     print(url)
#     rq = urllib2.Request(url)
#     res = urllib2.urlopen(rq)
#     pdf = open("pdfs/" + name, 'wb')
#     pdf.write(res.read())
#     pdf.close()