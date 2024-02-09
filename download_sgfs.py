import requests
from bs4 import BeautifulSoup as bs
from urllib.request import Request, urlopen
import os
import zipfile
import shutil
from sgf_to_position import process_selfplay

# already_analysed = ['b15c192-s100342528-d74656514', 'b15c192-s104075264-d75404031', 'b15c192-s110063872-d76577596', 'b15c192-s112036608-d76942297', 'b15c192-s113093120-d77167297', 'b15c192-s114148608-d77384942', 'b15c192-s116823552-d77817876', 'b15c192-s117739264-d78042876', 'b15c192-s118727424-d78194950', 'b15c192-s119643136-d78374439', 'b15c192-s121615104-d78701482', 'b15c192-s123517440-d79049862', 'b15c192-s74759936-d68801237', 'b15c192-s75747584-d69325930', 'b15c192-s77791744-d69831672', 'b15c192-s78778368-d70131672', 'b15c192-s79835136-d70391659', 'b15c192-s80751872-d70656711', 'b15c192-s85612288-d71992807', 'b15c192-s86740736-d72259836', 'b15c192-s87729152-d72306050', 'b15c192-s88713216-d72516067', 'b15c192-s90757120-d72874752', 'b15c192-s91673088-d73049752', 'b15c192-s93576704-d73351344', 'b15c192-s98580736-d74299748']

ALREADY_ANALYSED = "analysed_list.log"
OUTPUT_ZIP_DIR = "zip_logs_new"

if os.path.exists(ALREADY_ANALYSED):
    with open("analysed_list.log", "r") as f:
        text = f.read()
        already_analysed = text.split(" ")
else:
    already_analysed = []


_URL = "https://katagoarchive.org/g170/selfplay/index.html"
PROJECT_ROOT = os.getcwd()
selfplay_dir = os.path.join(PROJECT_ROOT, "kata-selfplay")
refined_log_dir = os.path.join(PROJECT_ROOT, "refined_logs")
log_dir = os.path.join(PROJECT_ROOT, "analysis_logs")
for d in [selfplay_dir, refined_log_dir, log_dir, OUTPUT_ZIP_DIR]:
    os.makedirs(d, exist_ok=True)

# functional
r = requests.get(_URL)
soup = bs(r.text)
urls = []
names = []
_URL_PARENT = _URL.rsplit('/', 1)[0]
used_counters = [int(a[:-4]) for a in os.listdir(OUTPUT_ZIP_DIR) if a.endswith('.zip')]
print(used_counters)
max_counter = 0 if len(used_counters) == 0 else max(used_counters)
print(max_counter)
counter = 20 * max_counter
for i, link in enumerate(soup.findAll('a')):
    ZIP_NAME = link.get('href')[2:]  # The first characters are a dot and a slash
    _FULLURL = _URL_PARENT + '/' + ZIP_NAME
    # print(_FULLURL)
    if ZIP_NAME.endswith('.zip') and ZIP_NAME[0] == 'b' and ZIP_NAME[1] != '6' and ZIP_NAME[1:3] != '10' and not ZIP_NAME[:-4] in already_analysed:
        try:
            counter += 1
            if counter % 20 == 0:
                shutil.make_archive(f"{OUTPUT_ZIP_DIR}/{counter//20:04}", "zip", refined_log_dir)
                shutil.rmtree(refined_log_dir)
                os.makedirs(refined_log_dir, exist_ok=True)
            zip_path = os.path.join(selfplay_dir, ZIP_NAME)
            refined_log_path = os.path.join(refined_log_dir, ZIP_NAME[:-4])

            if not os.path.exists(zip_path) and not os.path.exists(zip_path[:-4]):
                print(f'Downloading zip {ZIP_NAME}')
                rq = Request(url=_FULLURL,
                             headers={'User-Agent': 'Mozilla/5.0'})
                try:
                    zip_opened = urlopen(rq)
                except:
                    continue
                # res = urllib.urlopen(rq)
                # urls.append(_FULLURL)
                # names.append(soup.select('a')[i].attrs['href'])
                with open(zip_path, 'wb') as f:
                    f.write(zip_opened.read())
            if not os.path.exists(zip_path[:-4]):
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(members=[member for member in zip_ref.infolist() if 'sgfs' in str(member)],
                                       path=selfplay_dir)
                os.unlink(zip_path)
            if not os.path.exists(refined_log_path):
                print(f'Extracting zip {ZIP_NAME} and processing selfplay')
                process_selfplay(ZIP_NAME[:-4])
                # assert os.path.exists(refined_log_path)
                shutil.rmtree(zip_path[:-4], ignore_errors=True)
            already_analysed.append(ZIP_NAME[:-4])
        except Exception as e:
            raise e
        finally:
            with open("analysed_list.log", "w") as f:
                f.write(' '.join(already_analysed))

# names_urls = zip(names, urls)

# for name, url in names_urls:
#     print(url)
#     rq = urllib2.Request(url)
#     res = urllib2.urlopen(rq)
#     pdf = open("pdfs/" + name, 'wb')
#     pdf.write(res.read())
#     pdf.close()