import requests
from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import json
from PIL import Image
from io import BytesIO
import pytesseract
import time
import logging
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Create a logger object
logger = logging.getLogger()
# Create a filter to ignore specific warnings
class IgnoreWarningsFilter(logging.Filter):
    def filter(self, record):
        # Ignore specific warnings
        return not ("cannot identify image file" in record.getMessage())
logger.addFilter(IgnoreWarningsFilter())

def get_links(url):
    """
    Retrieve all the links from a given URL.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        for link in soup.find_all('a', href=True):
            absolute_link = urljoin(url, link.get('href'))
            parsed_link = urlparse(absolute_link)
            # Normalize by removing fragments and queries
            normalized_link = parsed_link._replace(fragment='', query='').geturl()
            links.add(normalized_link)
        return list(links)
    except RequestException as e:
        logging.error(f"Error retrieving links from {url}: {e}")
        return []

def filter_links(links, main_domain):
    valid_links = set()
    for link in links:
        parsed_url = urlparse(link)
        netloc = parsed_url.netloc
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        if parsed_url.scheme not in ('http', 'https'):
            continue
        if netloc != main_domain:
            continue
        if parsed_url.path.lower().endswith(('.jpg', '.png', '.gif', '.mp4', '.avi', '.mp3')):
            continue
        # Remove fragment and query for consistency
        normalized_url = parsed_url._replace(fragment='', query='').geturl()
        valid_links.add(normalized_url)
    return list(valid_links)

def perform_ocr_on_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        text = pytesseract.image_to_string(img)
        return text
    except RequestException as e:
        logging.error(f"Error retrieving image from {image_url}: {e}")
        return ""
    except Exception as e:
        logging.error(f"Error processing image {image_url}: {e}")
        return ""

def extract_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract Title
        title = soup.title.string.strip() if soup.title else None
        # Extract Headings and Paragraphs
        headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        # Extract Images with alt text and perform OCR
        images = []
        ocr_texts = []
        for img in soup.find_all('img', src=True):
            img_src = img['src']
            img_alt = img.get('alt', '')
            absolute_img_url = urljoin(url, img_src)
            ocr_text = perform_ocr_on_image(absolute_img_url)
            images.append({'src': img_src, 'alt': img_alt})
            if ocr_text.strip():
                ocr_texts.append(ocr_text)
        # Extract Metadata (description, keywords)
        metadata = {
            "description": soup.find('meta', attrs={'name': 'description'})['content'].strip() if soup.find('meta', attrs={'name': 'description'}) else None,
            "keywords": soup.find('meta', attrs={'name': 'keywords'})['content'].strip() if soup.find('meta', attrs={'name': 'keywords'}) else None,
        }
        # Extract JSON-LD Structured Data
        json_ld = []
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                json_ld.append(json.loads(script.string))
            except json.JSONDecodeError:
                continue
        # Extract other Structured Data (Microdata, RDFa)
        structured_data = []
        for tag in soup.find_all(True):
            if tag.has_attr('itemscope'):
                data = {}
                for prop in tag.find_all(True):
                    if prop.has_attr('itemprop'):
                        data[prop['itemprop']] = prop.get_text(strip=True)
                if data:
                    structured_data.append(data)
        # Extract Hidden Content (e.g., data in attributes or display:none)
        hidden_content = [hidden.get_text(strip=True) for hidden in soup.find_all(style=lambda x: x and 'display:none' in x)]
        content = {
            "title": title,
            "headings": headings,
            "paragraphs": paragraphs,
            "images": images,
            "metadata": metadata,
            "json_ld": json_ld,
            "structured_data": structured_data,
            "hidden_content": hidden_content,
            "ocr_texts": ocr_texts
        }
        return content
    except RequestException as e:
        logging.error(f"Error retrieving content from {url}: {e}")
        return {}
    except Exception as e:
        logging.error(f"Error processing content from {url}: {e}")
        return {}

def scrape_website(url, depth, main_domain, visited=None):
    if visited is None:
        visited = set()
    if depth == 0:
        logging.debug(f"Depth 0 reached for URL: {url}")
        return [{"url": url, "content": extract_content(url)}]
    if url in visited:
        logging.debug(f"URL already visited: {url}")
        return []
    logging.info(f"Scraping URL: {url} at depth: {depth}")
    visited.add(url)
    collected_data = [{"url": url, "content": extract_content(url)}]
    links = get_links(url)
    filtered_links = filter_links(links, main_domain)
    for link in filtered_links:
        if link in visited:
            logging.debug(f"Skipping already visited link: {link}")
            continue
        time.sleep(1)  # Respectful scraping
        collected_data.extend(scrape_website(link, depth - 1, main_domain, visited))
    return collected_data

def scrape_urls(website, depth=2):
    parsed_main = urlparse(website)
    main_domain = parsed_main.netloc
    if main_domain.startswith('www.'):
        main_domain = main_domain[4:]
    data = scrape_website(website, depth, main_domain)
    return data