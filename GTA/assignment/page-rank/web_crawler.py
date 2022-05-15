import requests
import lxml.html
import ssl
import certifi
from urllib.request import Request, urlopen

# request = "https://iitj.ac.in/"
# print(urlopen(request, context=ssl.create_default_context(cafile=certifi.where())))

url = "https://iitj.ac.in/"
headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,'
              'application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Cookie': 'PHPSESSID=rqq09vq3h157971if97l94u610; _ga=GA1.1.1316880827.1649409054; '
              '_ga_VWQ7ST9E2N=GS1.1.1649487675.4.0.1649487675.0',
    'Host': 'iitj.ac.in',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100", "Google Chrome";v="100"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/100.0.4896.75 Safari/537',
}

req = Request('http://api.company.com/items/details?country=US&language=en')
req.add_header('Accept-Encoding', 'gzip, deflate, br')
req.add_header('Accept-Language', 'en-GB,en-US;q=0.9,en;q=0.8')
req.add_header('Cache-Control', 'max-age=0')
req.add_header('Connection', 'keep-alive')
req.add_header('Cookie', 'PHPSESSID=rqq09vq3h157971if97l94u610; _ga=GA1.1.1316880827.1649409054; '
                         '_ga_VWQ7ST9E2N=GS1.1.1649487675.4.0.1649487675.0')
req.add_header('Host', 'iitj.ac.in')
req.add_header('sec-ch-ua', '" Not A;Brand";v="99", "Chromium";v="100", "Google Chrome";v="100"')
req.add_header('sec-ch-ua-mobile', '?0')
req.add_header('sec-ch-ua-platform', '"macOS"')
req.add_header('Sec-Fetch-Dest', 'document')
req.add_header('Sec-Fetch-Mode', 'navigate')
req.add_header('Sec-Fetch-Site', 'none')
req.add_header('Sec-Fetch-User', '?1')
req.add_header('Upgrade-Insecure-Requests', '1')
req.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) '
                             'Chrome/100.0.4896.75 Safari/537')
req.add_header('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,'
                         '*/*;q=0.8,application/signed-exchange;v=b3;q=0.9')

print(urlopen(url, context=ssl.create_default_context(cafile=certifi.where())))
