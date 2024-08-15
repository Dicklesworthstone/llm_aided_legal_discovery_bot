import asyncio
import httpx
from bs4 import BeautifulSoup
import os
import logging
from urllib.parse import urljoin
from tqdm.asyncio import tqdm
from httpx import HTTPStatusError, RequestError, TimeoutException

project_root = os.path.dirname(os.path.abspath(__file__))
original_source_dir = os.path.join(project_root, 'folder_of_source_documents__original_format')

# Setup logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('enron_downloader.log'), logging.StreamHandler()]
)

# Lower the verbosity of specific loggers
httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.WARNING)

# Step 1: Async function to retrieve a webpage with error handling
async def fetch_page(client, url, retries=3):
    for attempt in range(retries):
        try:
            response = await client.get(url, timeout=30)  # Increased timeout
            response.raise_for_status()
            return BeautifulSoup(response.text, 'lxml')
        except (HTTPStatusError, RequestError, TimeoutException) as e:
            logging.error(f"Error fetching {url} on attempt {attempt + 1}: {e}")
            if attempt + 1 == retries:
                logging.critical(f"Failed to fetch {url} after {retries} attempts.")
                return None

# Step 2: Async function to gather all exhibit URLs with error handling
async def gather_exhibit_urls(main_url, client):
    soup = await fetch_page(client, main_url)
    if soup is None:
        return []
    exhibit_urls = [
        urljoin(main_url, link['href'])
        for link in soup.find_all('a', href=True)
        if 'enron/exhibit/' in link['href']
    ]
    logging.info(f"Found {len(exhibit_urls)} exhibit URLs.")
    return exhibit_urls

# Step 3: Async function to gather all PDF URLs from exhibit pages
async def gather_pdf_urls(exhibit_urls, client):
    pdf_urls = set()  # Use a set to avoid duplicates
    for exhibit_url in exhibit_urls:
        logging.info(f"Processing exhibit index page: {exhibit_url}")
        exhibit_soup = await fetch_page(client, exhibit_url)
        if exhibit_soup is None:
            continue
        exhibit_pdf_urls = []
        
        # Find all 'a' tags that contain PDF links
        for a_tag in exhibit_soup.find_all('a', href=True):
            if a_tag['href'].lower().endswith('.pdf'):
                full_url = urljoin(exhibit_url, a_tag['href'])
                exhibit_pdf_urls.append(full_url)
                pdf_urls.add(full_url)
                logging.debug(f"Found PDF URL: {full_url}")
        
        logging.info(f"Found {len(exhibit_pdf_urls)} PDF URLs on exhibit index page: {exhibit_url}")
    
    logging.info(f"Collected {len(pdf_urls)} unique PDF URLs in total.")
    return list(pdf_urls)

# Step 4: Async function to download a PDF with a semaphore limit and error handling
async def download_pdf(sem, client, pdf_url, pbar, retries=3):
    async with sem:
        pdf_name = os.path.basename(pdf_url)
        pdf_path = os.path.join(original_source_dir, pdf_name)
        
        if os.path.exists(pdf_path):
            logging.info(f"PDF already exists: {pdf_path}. Skipping download.")
            pbar.update(1)
            return

        for attempt in range(retries):
            try:
                pbar.set_description(f"Downloading: {pdf_url}")
                response = await client.get(pdf_url, timeout=60)  # Increase timeout
                response.raise_for_status()
                with open(pdf_path, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                pbar.update(1)
                logging.info(f"Successfully downloaded {pdf_url} to {pdf_path}")
                break
            except (HTTPStatusError, RequestError, TimeoutException) as e:
                logging.error(f"Error downloading {pdf_url} on attempt {attempt + 1}: {e}")
                if attempt + 1 == retries:
                    logging.critical(f"Failed to download {pdf_url} after {retries} attempts.")

# Step 5: Main async function to run all tasks
async def main():
    main_url = 'https://www.justice.gov/archive/index-enron.html'
    sem = asyncio.Semaphore(5)
    
    # Ensure the download directory exists
    os.makedirs(original_source_dir, exist_ok=True)
    
    # Check the number of PDF files already downloaded
    expected_number_of_pdfs = 950
    existing_pdfs = [
        f for f in os.listdir(original_source_dir)
        if os.path.isfile(os.path.join(original_source_dir, f)) and f.lower().endswith('.pdf') and os.path.getsize(os.path.join(original_source_dir, f)) > 25 * 1024
    ]
    
    if len(existing_pdfs) >= expected_number_of_pdfs:
        logging.info(f"Found {len(existing_pdfs)} PDF files of size at least 25KB in {original_source_dir}.")
        logging.info("Looks like the files have already been downloaded. Exiting.")
        return
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=60.0)) as client:
        exhibit_urls = await gather_exhibit_urls(main_url, client)
        if not exhibit_urls:
            logging.error("No exhibit URLs found. Exiting.")
            return

        pdf_urls = await gather_pdf_urls(exhibit_urls, client)
        if not pdf_urls:
            logging.error("No PDF URLs found. Exiting.")
            return
        
        # Write the list of PDF URLs to a file
        with open('list_of_enron_exhibit_pdf_urls.txt', 'w') as file:
            for pdf_url in pdf_urls:
                file.write(pdf_url + '\n')

        # Download all PDFs concurrently with a progress bar
        pbar = tqdm(total=len(pdf_urls), desc="Processing PDFs")
        tasks = [download_pdf(sem, client, pdf_url, pbar) for pdf_url in pdf_urls]
        await asyncio.gather(*tasks)
        pbar.close()

# Step 6: Run the main function
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        logging.critical(f"Script terminated with an unexpected error: {e}")
