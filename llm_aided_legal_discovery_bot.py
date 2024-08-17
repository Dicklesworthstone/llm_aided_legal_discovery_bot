import os
import glob
import traceback
import asyncio
import aiofiles
import time
import multiprocessing
from functools import partial
from aiolimiter import AsyncLimiter
import json
import re
import io
import shutil
import base64
import sqlite3
import zipfile
import hashlib
import warnings
from email.utils import parseaddr, parsedate_to_datetime
from email.parser import BytesParser
from email.policy import default
from collections import Counter, defaultdict
import urllib.request
from typing import List, Dict, Tuple, Optional, Set, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import picologging as logging
from decouple import Config as DecoupleConfig, RepositoryEnv
from magika import Magika
from tqdm import tqdm
import httpx
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential
import textract
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2
import pypff
from llama_cpp import Llama, LlamaGrammar
import tiktoken
from filelock import FileLock, Timeout
from transformers import AutoTokenizer
from openai import AsyncOpenAI, APIError, RateLimitError
from anthropic import AsyncAnthropic
from enron_sample_data_collector_script import main as enron_collector_main

try:
    import nvgpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Configuration
config = DecoupleConfig(RepositoryEnv('.env'))

# Global variables controlling the LLM API/model behavior
USE_LOCAL_LLM = config.get("USE_LOCAL_LLM", default=False, cast=bool)
API_PROVIDER = config.get("API_PROVIDER", default="OPENAI", cast=str) # OPENAI or CLAUDE
ANTHROPIC_API_KEY = config.get("ANTHROPIC_API_KEY", default="your-anthropic-api-key", cast=str)
OPENAI_API_KEY = config.get("OPENAI_API_KEY", default="your-openai-api-key", cast=str)
CLAUDE_MODEL_STRING = config.get("CLAUDE_MODEL_STRING", default="claude-3-haiku-20240307", cast=str)
CLAUDE_MAX_TOKENS = 4096 # Maximum allowed tokens for Claude API
TOKEN_BUFFER = 500  # Buffer to account for token estimation inaccuracies
TOKEN_CUSHION = 300 # Don't use the full max tokens to avoid hitting the limit
OPENAI_COMPLETION_MODEL = config.get("OPENAI_COMPLETION_MODEL", default="gpt-4o-mini", cast=str)
OPENAI_EMBEDDING_MODEL = config.get("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small", cast=str)
OPENAI_MAX_TOKENS = 128000  # Context window for GPT-4o mini
DEFAULT_LOCAL_MODEL_NAME = "Llama-3.1-8B-Lexi-Uncensored_Q5_fixedrope.gguf"
LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS = 2048
USE_VERBOSE = False

magika = Magika() # Initialize Magika
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) # Initialize OpenAI client
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'  # ISO format
)

# Create a rate limiter for API requests
rate_limit = AsyncLimiter(max_rate=60, time_period=60)  # 60 requests per minute

def remove_pagination_breaks(text: str) -> str:
    text = re.sub(r'-(\n)(?=[a-z])', '', text)  # Remove hyphens at the end of lines when the word continues on the next line
    text = re.sub(r'(?<=\w)(?<![.?!-]|\d)\n(?![\nA-Z])', ' ', text)  # Replace line breaks that are not preceded by punctuation or list markers and not followed by an uppercase letter or another line break   
    return text

def sophisticated_sentence_splitter(text: str) -> List[str]:
    text = remove_pagination_breaks(text)
    pattern = r'\.(?!\s*(com|net|org|io)\s)(?![0-9])'  # Split on periods that are not followed by a space and a top-level domain or a number
    pattern += r'|[.!?]\s+'  # Split on whitespace that follows a period, question mark, or exclamation point
    pattern += r'|\.\.\.(?=\s)'  # Split on ellipses that are followed by a space
    sentences = re.split(pattern, text)
    refined_sentences = []
    temp_sentence = ""
    for sentence in sentences:
        if sentence is not None:
            temp_sentence += sentence
            if temp_sentence.count('"') % 2 == 0:  # If the number of quotes is even, then we have a complete sentence
                refined_sentences.append(temp_sentence.strip())
                temp_sentence = ""
    if temp_sentence:
        if refined_sentences:
            refined_sentences[-1] += " " + temp_sentence.strip()
        else:
            refined_sentences.append(temp_sentence.strip())
    return [s.strip() for s in refined_sentences if s.strip()]

async def parse_email_async(file_path: str) -> Dict[str, Any]:
    async with aiofiles.open(file_path, 'rb') as file:
        content = await file.read()
    msg = BytesParser(policy=default).parsebytes(content)
    headers = {
        'From': msg['from'],
        'To': msg['to'],
        'Subject': msg['subject'],
        'Date': msg['date']
    }
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                break
    else:
        body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
    return {
        'headers': headers,
        'body': body
    }

def minimal_clean_email_body(body: str) -> str:
    # Step 1: Normalize line breaks
    body = body.replace('\r\n', '\n').replace('\r', '\n')
    # Step 2: Remove any null bytes
    body = body.replace('\x00', '')
    # Step 3: Remove excessive blank lines (more than 3 in a row); This preserves intentional spacing while removing only extreme cases
    body = re.sub(r'\n{4,}', '\n\n\n', body)
    # Step 4: Ensure the email ends with a single newline
    body = body.rstrip() + '\n'
    return body

async def parse_enron_email_async(file_path: str) -> Dict[str, Any]:
    async with aiofiles.open(file_path, 'rb') as file:
        content = await file.read()
    msg = BytesParser(policy=default).parsebytes(content)
    headers = {
        'From': msg['from'],
        'To': msg['to'],
        'Subject': msg['subject'],
        'Date': msg['date'],
        'Cc': msg['cc'],
        'Bcc': msg['bcc'],
        'X-Folder': msg['X-Folder'],
        'X-Origin': msg['X-Origin'],
        'X-FileName': msg['X-FileName'],
    }
    for key, value in headers.items():
        if value:
            headers[key] = ' '.join(str(value).split())
    from_name, from_email = parseaddr(headers['From'])
    headers['From'] = {'name': from_name, 'email': from_email}
    if headers['To']:
        headers['To'] = [{'name': name, 'email': email} for name, email in [parseaddr(addr) for addr in headers['To'].split(',')]]
    else:
        headers['To'] = []
    for field in ['Cc', 'Bcc']:
        if headers[field]:
            headers[field] = [{'name': name, 'email': email} for name, email in [parseaddr(addr) for addr in str(headers[field]).split(',')]]
        else:
            headers[field] = []
    if headers['Date']:
        try:
            headers['Date'] = parsedate_to_datetime(headers['Date']).isoformat()
        except:  # noqa: E722
            pass
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
    else:
        body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
    body = minimal_clean_email_body(body) # Apply the minimal cleanup function
    return {
        'headers': headers,
        'body': body
    }

async def parse_document_into_sentences(file_path: str, mime_type: str) -> Tuple[List[str], float, Dict[str, Any]]:
    content = ""
    email_metadata = {}
    if mime_type == 'message/rfc822':  # This is an email file
        email_content = parse_email_async(file_path)
        email_metadata = email_content['headers']
        content = f"From: {email_metadata['From']}\nTo: {email_metadata['To']}\nSubject: {email_metadata['Subject']}\nDate: {email_metadata['Date']}\n\n{email_content['body']}"
    else:
        try:
            content = textract.process(file_path, encoding='utf-8')
            content = content.decode('utf-8')
        except Exception as e:
            logging.error(f"Error while processing file: {e}, mime_type: {mime_type}")
            logging.error(traceback.format_exc())
            raise ValueError(f"Unsupported file type or error: {e}")
    sentences = sophisticated_sentence_splitter(content)
    if len(sentences) == 0 and file_path.lower().endswith('.pdf'):
        logging.info("No sentences found, attempting OCR using Tesseract.")
        try:
            content = textract.process(file_path, method='tesseract', encoding='utf-8')
            content = content.decode('utf-8')
            sentences = sophisticated_sentence_splitter(content)
        except Exception as e:
            logging.error(f"Error while processing file with OCR: {e}")
            logging.error(traceback.format_exc())
            raise ValueError(f"OCR failed: {e}")
    if len(sentences) == 0:
        logging.info("No sentences found in the document")
        raise ValueError("No sentences found in the document")
    strings = [s.strip() for s in sentences]
    thousands_of_input_words = round(sum(len(s.split()) for s in strings) / 1000, 2)
    return strings, thousands_of_input_words, email_metadata

async def download_and_extract_enron_emails_dataset(url: str, destination_folder: str):
    zip_file_path = os.path.join(destination_folder, "enron_dataset.zip")
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    # Download the file if it doesn't exist
    if not os.path.exists(zip_file_path):
        logging.info(f"Downloading Enron dataset from {url}")
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                with open(zip_file_path, "wb") as file, tqdm(
                    desc="Downloading Enron dataset",
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress_bar:
                    async for chunk in response.aiter_bytes():
                        size = file.write(chunk)
                        progress_bar.update(size)
    else:
        logging.info(f"Enron dataset zip file already exists at {zip_file_path}")
    # Extract the ZIP file
    logging.info("Extracting Enron dataset...")
    temp_extract_folder = os.path.join(destination_folder, "temp_extract")
    os.makedirs(temp_extract_folder, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_folder)
    # Move the maildir folder to the correct location
    extracted_maildir = os.path.join(temp_extract_folder, '2018487913', 'maildir')
    final_maildir = os.path.join(destination_folder, 'maildir')
    if os.path.exists(extracted_maildir):
        if os.path.exists(final_maildir):
            shutil.rmtree(final_maildir)
        shutil.move(extracted_maildir, final_maildir)
        logging.info(f"Maildir moved to: {final_maildir}")
    else:
        logging.error(f"Maildir not found in the extracted dataset at {extracted_maildir}")
    # Clean up
    shutil.rmtree(temp_extract_folder)
    os.remove(zip_file_path)
    logging.info("Enron dataset extracted and cleaned up")
    if os.path.exists(final_maildir):
        return final_maildir
    else:
        logging.error("Failed to locate the final Maildir")
        return None

async def process_extracted_enron_emails(maildir_path: str, converted_source_dir: str):
    logging.info(f"Now processing extracted Enron email corpus from {maildir_path}")
    semaphore = asyncio.Semaphore(25000)
    async def process_subfolder(sender: str, subfolder: str, subfolder_path: str, pbar: tqdm):
        safe_sender = ''.join(c.lower() if c.isalnum() else '_' for c in sender)
        safe_subfolder = ''.join(c.lower() if c.isalnum() else '_' for c in subfolder)
        output_file_name = f"email_bundle__sender__{safe_sender}__category__{safe_subfolder}.md"
        output_file_path = os.path.join(converted_source_dir, output_file_name)
        if os.path.exists(output_file_path):
            logging.info(f"Skipping already processed bundle: {output_file_name}")
            pbar.update(sum(1 for _ in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, _))))
            return
        emails = []
        async def process_file(file_path: str):
            async with semaphore:
                try:
                    email_data = await parse_enron_email_async(file_path)
                    email_data['file_path'] = file_path
                    headers_str = json.dumps(email_data['headers'], sort_keys=True)
                    content_for_hash = headers_str + email_data['body']
                    email_unique_identifier = hashlib.sha256(content_for_hash.encode()).hexdigest()[:16]
                    email_data['unique_identifier'] = email_unique_identifier
                    emails.append(email_data)
                    pbar.set_postfix_str(f"Processed: {os.path.basename(file_path)}", refresh=False)
                    pbar.update(1)
                except Exception as e:
                    pbar.set_postfix_str(f"Error: {os.path.basename(file_path)}", refresh=False)
                    pbar.update(1)
                    logging.error(f"Error processing {file_path}: {str(e)}")
        tasks = []
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            if os.path.isfile(file_path):
                tasks.append(asyncio.create_task(process_file(file_path)))
        await asyncio.gather(*tasks)
        if emails:
            markdown_content = ""
            for index, email_data in enumerate(emails, 1):
                markdown_content += f"# Email {index} of {len(emails)} from {sender} in {subfolder}\n\n"
                markdown_content += f"**Unique Email Identifier:** {email_data['unique_identifier']}\n"
                markdown_content += f"**From:** {email_data['headers']['From']['name']} <{email_data['headers']['From']['email']}>\n"
                markdown_content += f"**To:** {', '.join([f'{r['name']} <{r['email']}>' for r in email_data['headers']['To']])}\n"
                markdown_content += f"**Subject:** {email_data['headers']['Subject']}\n"
                markdown_content += f"**Date:** {email_data['headers']['Date']}\n"
                if email_data['headers']['Cc']:
                    markdown_content += f"**Cc:** {', '.join([f'{r['name']} <{r['email']}>' for r in email_data['headers']['Cc']])}\n"
                if email_data['headers']['Bcc']:
                    markdown_content += f"**Bcc:** {', '.join([f'{r['name']} <{r['email']}>' for r in email_data['headers']['Bcc']])}\n"
                markdown_content += f"**X-Folder:** {email_data['headers']['X-Folder']}\n"
                markdown_content += f"**X-Origin:** {email_data['headers']['X-Origin']}\n"
                markdown_content += f"**X-FileName:** {email_data['headers']['X-FileName']}\n\n"
                markdown_content += f"**Body:**\n\n{email_data['body']}\n\n"
                markdown_content += "---\n\n"
            os.makedirs(converted_source_dir, exist_ok=True)
            async with aiofiles.open(output_file_path, 'w', encoding='utf-8') as f:
                await f.write(markdown_content)
            logging.info(f"Created markdown file: {output_file_name}")
    # Get total number of files
    total_files = sum([len(files) for _, _, files in os.walk(maildir_path)])
    with tqdm(total=total_files, desc="Processing Enron emails", unit="email") as pbar:
        for sender in os.listdir(maildir_path):
            sender_path = os.path.join(maildir_path, sender)
            if os.path.isdir(sender_path):
                for subfolder in os.listdir(sender_path):
                    subfolder_path = os.path.join(sender_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        await process_subfolder(sender, subfolder, subfolder_path, pbar)
    logging.info("Finished processing Enron email corpus.")
    
async def process_enron_email_corpus(
    project_root: str,
    original_source_dir: str,
    converted_source_dir: str
):
    logging.info("Processing Enron email corpus")
    enron_dataset_url = "https://tile.loc.gov/storage-services/master/gdc/gdcdatasets/2018487913/2018487913.zip"                
    enron_dataset_dir = os.path.join(project_root, 'enron_email_data')
    os.makedirs(enron_dataset_dir, exist_ok=True)
    maildir_path = os.path.join(enron_dataset_dir, 'maildir')
    # Check if the maildir already contains the expected number of subdirectories
    if os.path.exists(maildir_path):
        subdirs = [d for d in os.listdir(maildir_path) if os.path.isdir(os.path.join(maildir_path, d))]
        if len(subdirs) == 150:
            logging.info("Enron email corpus already extracted and present. Skipping download and extraction.")
            return await process_extracted_enron_emails(maildir_path, converted_source_dir)
    # If we don't have the complete extracted data, proceed with download and extraction
    logging.info("Downloading and extracting Enron dataset...")
    maildir_path = await download_and_extract_enron_emails_dataset(enron_dataset_url, enron_dataset_dir)
    if not maildir_path or not os.path.exists(maildir_path):
        logging.error("Failed to download or extract Enron dataset. Skipping Enron email processing.")
        return
    return await process_extracted_enron_emails(maildir_path, converted_source_dir)


##########################################################################
# OCR Helper Functions
##########################################################################

def robust_needs_ocr(file_path: str) -> bool:
    """
    Determine if a file needs OCR using multiple methods.
    """
    logging.info(f"Checking if {file_path} needs OCR")
    # For PDFs
    if file_path.lower().endswith('.pdf'):
        return needs_ocr_pdf(file_path)
    # For images
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        return True
    # For other file types, assume no OCR needed
    return False

def needs_ocr_pdf(pdf_path: str) -> bool:
    """
    Check if a PDF needs OCR by attempting multiple methods of text extraction.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # Check a sample of pages (e.g., first, middle, and last if available)
            pages_to_check = [0, len(reader.pages) // 2, -1]
            for page_num in pages_to_check:
                if page_num < len(reader.pages):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    # If we find meaningful text, no OCR needed
                    if len(text.strip()) > 50:  # Adjust threshold as needed
                        logging.info(f"Found sufficient text in PDF {pdf_path} on page {page_num+1}. OCR not needed.")
                        return False
    except Exception as e:
        logging.error(f"Error checking PDF {pdf_path}: {str(e)}")
    logging.info(f"PDF {pdf_path} likely needs OCR.")
    return True

def resample_to_300_dpi(image: Image.Image) -> Image.Image:
    dpi = 300
    width_inch, height_inch = image.size[0] / dpi, image.size[1] / dpi
    resampled_image = image.resize((int(width_inch * dpi), int(height_inch * dpi)), Image.LANCZOS)
    return resampled_image

def perform_simple_ocr_on_image(image_path: str) -> str:
    logging.info(f"Performing OCR on image: {image_path}")
    image = Image.open(image_path)
    resampled_image = resample_to_300_dpi(image)
    # Convert to grayscale and prepare the image for OCR
    gray = cv2.cvtColor(np.array(resampled_image), cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # Apply dilation to connect text components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.dilate(gray, kernel, iterations=1)
    # Perform text extraction
    text = pytesseract.image_to_string(gray)
    return text

def preprocess_image_ocr(image):
    # Resample the image to 300 dpi (tesseract was trained on this resolution)
    resampled_image = resample_to_300_dpi(image)
    gray = cv2.cvtColor(np.array(resampled_image), cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    return Image.fromarray(gray)

def convert_pdf_to_images_ocr(input_pdf_file_path: str, max_pages: int = 0, skip_first_n_pages: int = 0) -> List[Image.Image]:
    logging.info(f"Processing PDF file {input_pdf_file_path}")
    if max_pages == 0:
        last_page = None
        logging.info("Converting all pages to images...")
    else:
        last_page = skip_first_n_pages + max_pages
        logging.info(f"Converting pages {skip_first_n_pages + 1} to {last_page}")
    first_page = skip_first_n_pages + 1  # pdf2image uses 1-based indexing
    images = convert_from_path(input_pdf_file_path, first_page=first_page, last_page=last_page)
    logging.info(f"Converted {len(images)} pages from PDF file to images.")
    return images

def ocr_image(image):
    preprocessed_image = preprocess_image_ocr(image)
    return pytesseract.image_to_string(preprocessed_image)

def escape_special_characters(text):
    return text.replace('{', '{{').replace('}', '}}')

async def process_text_with_llm(text: str, prompt_template: str, max_chunk_size: int = 1000, prev_context: str = "", escape_text: bool = True) -> str:
    if escape_text:
        text = escape_special_characters(text)
    model_name = OPENAI_COMPLETION_MODEL if API_PROVIDER == "OPENAI" else CLAUDE_MODEL_STRING
    async def process_chunk(chunk: str) -> str:
        prompt = prompt_template.format(text=chunk, prev_context=prev_context)
        max_tokens = min(2048, 4096 - estimate_tokens(prompt, model_name))
        result = await generate_completion(prompt, max_tokens=max_tokens)
        return result if result is not None else chunk
    if estimate_tokens(text, model_name) <= max_chunk_size:
        return await process_chunk(text)
    parts = []
    words = text.split()
    current_part = []
    current_tokens = 0
    for word in words:
        word_tokens = estimate_tokens(word, model_name)
        if current_tokens + word_tokens > max_chunk_size:
            part_text = " ".join(current_part)
            processed_part = await process_chunk(part_text)
            parts.append(processed_part)
            current_part = [word]
            current_tokens = word_tokens
        else:
            current_part.append(word)
            current_tokens += word_tokens
    if current_part:
        part_text = " ".join(current_part)
        processed_part = await process_chunk(part_text)
        parts.append(processed_part)
    return " ".join(parts)

async def process_chunk_ocr(chunk: str, prev_context: str, chunk_index: int, total_chunks: int, reformat_as_markdown: bool, suppress_headers_and_page_numbers: bool) -> Tuple[str, str]:
    logging.info(f"Processing OCR chunk {chunk_index + 1}/{total_chunks} (length: {len(chunk):,} characters)")
    # Check if the chunk has enough content
    if len(chunk.strip()) < 50:
        logging.warning(f"Skipping chunk {chunk_index + 1}/{total_chunks} due to insufficient content (less than 50 characters)")
        return "", prev_context
    context_length = min(250, len(prev_context))
    prev_context_short = prev_context[-context_length:]
    # OCR correction prompt template
    ocr_correction_template = """Correct OCR-induced errors in the text, ensuring it flows coherently with the previous context. Follow these guidelines:

1. Fix OCR-induced typos and errors:
   - Correct words split across line breaks
   - Fix common OCR errors (e.g., 'rn' misread as 'm')
   - Use context and common sense to correct errors
   - Only fix clear errors, don't alter the content unnecessarily
   - Do not add extra periods or any unnecessary punctuation

2. Maintain original structure:
   - Keep all headings and subheadings intact

3. Preserve original content:
   - Keep all important information from the original text
   - Do not add any new information not present in the original text
   - Remove unnecessary line breaks within sentences or paragraphs
   - Maintain paragraph breaks
   
4. Maintain coherence:
   - Ensure the content connects smoothly with the previous context
   - Handle text that starts or ends mid-sentence appropriately

IMPORTANT: Respond ONLY with the corrected text. Preserve all original formatting, including line breaks. Do not include any introduction, explanation, or metadata.

Previous context:
{prev_context}

Current text to process:
{text}

Corrected text:
"""
    ocr_corrected_chunk = await process_text_with_llm(chunk, ocr_correction_template, prev_context=prev_context_short)
    processed_chunk = ocr_corrected_chunk
    if reformat_as_markdown:
        markdown_template = """Reformat the following text as markdown, improving readability while preserving the original structure. Follow these guidelines:
1. Preserve all original headings, converting them to appropriate markdown heading levels (# for main titles, ## for subtitles, etc.)
   - Ensure each heading is on its own line
   - Add a blank line before and after each heading
2. Maintain the original paragraph structure. Remove all breaks within a word that should be a single word (for example, "cor- rect" should be "correct")
3. Format lists properly (unordered or ordered) if they exist in the original text
4. Use emphasis (*italic*) and strong emphasis (**bold**) where appropriate, based on the original formatting
5. Preserve all original content and meaning
6. Do not add any extra punctuation or modify the existing punctuation
7. Do not add any introductory text, preamble, or markdown code block indicators
8. Remove any obviously duplicated content that appears to have been accidentally included twice. Follow these strict guidelines:
   - Remove only exact or near-exact repeated paragraphs or sections within the main chunk
   - Consider the context (before and after the main chunk) to identify duplicates that span chunk boundaries
   - Do not remove content that is simply similar but conveys different information
   - Preserve all unique content, even if it seems redundant
   - Ensure the text flows smoothly after removal
   - Do not add any new content or explanations
   - If no obvious duplicates are found, return the main chunk unchanged
9. {suppress_instruction}

IMPORTANT: Do not add any markdown code block indicators, preamble, or introductory text. Start directly with the reformatted content.

Text to reformat:

{text}

Reformatted markdown:
"""
        suppress_instruction = "Carefully remove headers, footers, and page numbers while preserving all other content." if suppress_headers_and_page_numbers else "Identify but do not remove headers, footers, or page numbers. Instead, format them distinctly, e.g., as blockquotes."
        markdown_template_formatted = markdown_template.format(suppress_instruction=suppress_instruction, text="{text}")
        processed_chunk = await process_text_with_llm(ocr_corrected_chunk, markdown_template_formatted, prev_context=prev_context_short)

    # Additional filtering stage
    filtering_template = """Review the following markdown-formatted text and remove any invalid or unwanted elements without altering the actual content. Follow these guidelines:

1. Remove any markdown code block indicators (```) if present
2. Remove any preamble or introductory text such as "Reformatted markdown:" or "Here is the corrected text:"
3. Ensure the text starts directly with the content (e.g., headings, paragraphs, or lists)
4. Do not remove any actual content, headings, or meaningful text
5. Preserve all markdown formatting (headings, lists, emphasis, etc.)
6. Remove any trailing whitespace or unnecessary blank lines at the end of the text

IMPORTANT: Only remove invalid elements as described above. Do not alter, summarize, or remove any of the actual content.

Text to filter:

{text}

Filtered text:
"""
    filtered_chunk = await process_text_with_llm(processed_chunk, filtering_template, prev_context=prev_context_short)
    # Check if the final output is reasonably long
    if len(filtered_chunk.strip()) < 100:
        logging.warning(f"Chunk {chunk_index + 1}/{total_chunks} output is too short (less than 100 characters). This may indicate a processing issue.")
        return chunk, prev_context  # Return original chunk if processed result is too short
    # Use dynamic context length for the next chunk
    new_context_length = min(500, len(filtered_chunk))
    new_context = filtered_chunk[-new_context_length:]
    logging.info(f"OCR Chunk {chunk_index + 1}/{total_chunks} processed. Output length: {len(filtered_chunk):,} characters")
    return filtered_chunk, new_context

async def process_document_ocr(list_of_extracted_text_strings: List[str], reformat_as_markdown: bool = True, suppress_headers_and_page_numbers: bool = True) -> str:
    logging.info(f"Starting OCR document processing. Total pages: {len(list_of_extracted_text_strings):,}")
    full_text = "\n\n".join(list_of_extracted_text_strings)
    logging.info(f"Size of full OCR text before processing: {len(full_text):,} characters")
    chunk_size, overlap = 8000, 10
    paragraphs = re.split(r'\n\s*\n', full_text)
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    # Chunking logic (unchanged)
    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_chunk_length + paragraph_length <= chunk_size:
            current_chunk.append(paragraph)
            current_chunk_length += paragraph_length
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            current_chunk = []
            current_chunk_length = 0
            for sentence in sentences:
                sentence_length = len(sentence)
                if current_chunk_length + sentence_length <= chunk_size:
                    current_chunk.append(sentence)
                    current_chunk_length += sentence_length
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_chunk_length = sentence_length
    if current_chunk:
        chunks.append("\n\n".join(current_chunk) if len(current_chunk) > 1 else current_chunk[0])
    # Add overlap between chunks
    for i in range(1, len(chunks)):
        overlap_text = chunks[i-1].split()[-overlap:]
        chunks[i] = " ".join(overlap_text) + " " + chunks[i]
    logging.info(f"OCR document split into {len(chunks):,} chunks. Chunk size: {chunk_size:,}, Overlap: {overlap:,}")
    # Concurrent processing of chunks
    semaphore = asyncio.Semaphore(150)  # Limit concurrent processing to 150 chunks at a time
    async def process_chunk_wrapper(chunk: str, index: int) -> str:
        async with semaphore:
            try:
                logging.info(f"Processing chunk {index+1}/{len(chunks)}")
                processed_chunk = await process_chunk_ocr(chunk, "", index, len(chunks), reformat_as_markdown, suppress_headers_and_page_numbers)
                logging.info(f"Chunk {index+1}/{len(chunks)} processed successfully")
                return processed_chunk[0]  # process_chunk_ocr returns a tuple, we want the first element
            except Exception as e:
                logging.error(f"Error processing chunk {index+1}/{len(chunks)}: {str(e)}")
                logging.error(traceback.format_exc())
                logging.warning(f"Using original chunk for {index+1}/{len(chunks)} due to processing error")
                return chunk
    # Use asyncio.gather to process chunks concurrently
    processed_chunks = await asyncio.gather(*[process_chunk_wrapper(chunk, i) for i, chunk in enumerate(chunks)])
    final_text = "".join(processed_chunks)
    logging.info(f"Size of OCR text after combining chunks: {len(final_text):,} characters")
    logging.info(f"OCR document processing complete. Final text length: {len(final_text):,} characters")
    return final_text

##########################################################################
# Legal Discovery Prompt Templates
##########################################################################

# Document Identification
doc_id_prompt = """
Analyze the following document excerpt and determine its nature. Consider the following aspects:
1. Document type (e.g., email, contract, report, memo)
2. Date of creation (if available)
3. Author and recipient (if applicable)
4. General subject matter

Document excerpt:
{document_excerpt}

Entities of Interest:
{entities_of_interest}

Provide your analysis in the following format (if a field is not applicable, leave it blank; if it's not available, use "N/A"):
TYPE: [Document Type]
DATE: [Date of Creation]
AUTHOR: [Author]
RECIPIENT: [Recipient]
SUBJECT: [Brief description of subject matter]
ENTITIES_MENTIONED: [List of entities of interest mentioned in the excerpt]
"""

# Relevance Check
relevance_check_prompt = """
Given the following discovery goals and document information, determine if this document is relevant to the case.

Discovery Goals:
{discovery_goals}

Document Information:
{doc_info}

Document excerpt:
{document_excerpt}

Keywords of Interest:
{keywords}

Entities of Interest:
{entities_of_interest}

Is this document relevant to any of the discovery goals? If so, which ones and why? If not, explain why.

Provide your analysis in the following format:
RELEVANT: [Yes/No]
GOALS: [List of relevant goal numbers]
KEYWORDS_FOUND: [List of keywords found in the excerpt]
ENTITIES_FOUND: [List of entities of interest found in the excerpt]
EXPLANATION: [Brief explanation of relevance or lack thereof, mentioning specific goals, keywords, and entities]
"""

# Extract Generation
extract_gen_prompt = """
Based on the relevance analysis, generate the most important extracts from the document that support its relevance to the discovery goals.

Relevance Analysis:
{relevance_analysis}

Full Document Text:
{full_document_text}

Discovery Goals:
{discovery_goals}

Keywords of Interest:
{keywords}

Entities of Interest:
{entities_of_interest}

Generate up to 3 key extracts, using ellipses (...) to focus on the most relevant parts. Use markdown formatting (italic and bold) to emphasize the most important words or phrases, especially the keywords and entities of interest, and in particular, how these pertain to the stated discovery goals.

Provide your extracts in the following format:
EXTRACT 1: [Extract with formatting]
EXTRACT 2: [Extract with formatting]
EXTRACT 3: [Extract with formatting]

For each extract, also provide:
RELEVANCE 1: [Brief explanation of how this extract relates to specific discovery goals]
RELEVANCE 2: [Brief explanation of how this extract relates to specific discovery goals]
RELEVANCE 3: [Brief explanation of how this extract relates to specific discovery goals]
"""

# Tag Generation
tag_gen_prompt = """
Based on the document information, relevance analysis, and key extracts, generate a set of tags (like #hashtags, using the hashtag symbol and only lowercase letters and underscores) that describe the content and its relevance to the discovery goals.

Document Information:
{doc_info}

Relevance Analysis:
{relevance_analysis}

Key Extracts:
{key_extracts}

Discovery Goals:
{discovery_goals}

Keywords of Interest:
{keywords}

Entities of Interest:
{entities_of_interest}

Generate up to 15 tags that best categorize this document and its relevance to the case. If fewer than 15 tags are suitable, include only the most relevant tags that would be useful for the discovery goals. Include tags for relevant discovery goals, keywords found, and entities mentioned.

Provide your tags in the following format:
TAGS: [tag1], [tag2], [tag3], [tag4], [tag5], [tag6], [tag7], [tag8], [tag9], [tag10], [tag11], [tag12], [tag13], [tag14], [tag15]
"""

# Explanation Generation
explanation_gen_prompt = """
Provide a comprehensive explanation of why this document is important or relevant to the discovery goals. Be specific and provide examples to support your explanation if applicable.

Discovery Goals:
{discovery_goals}

Document Information:
{doc_info}

Key Extracts:
{key_extracts}

Relevance Analysis:
{relevance_analysis}

Keywords Found:
{keywords_found}

Entities Mentioned:
{entities_mentioned}

Provide a detailed explanation (3-5 sentences) of the importance of this document. Address the following points:
1. How the document relates to specific discovery goals
2. The significance of any keywords or entities found
3. Potential impact on the case
4. Any unique or crucial information provided by this document

EXPLANATION: [Your explanation here]
"""

# Importance Score Generation
importance_score_prompt = """
Based on all the information gathered about this document, generate sub-scores for different aspects of importance to the discovery goals of the case. Consider the following factors:

Document Information:
{doc_info}

Relevance Analysis:
{relevance_analysis}

Key Extracts:
{key_extracts}

Tags:
{tags}

Explanation:
{explanation}

Discovery Goals:
{discovery_goals}

Keywords Found:
{keywords_found}

Entities Mentioned:
{entities_mentioned}

Please provide sub-scores for each of the following categories on a scale of 0 to 100:

1. Relevance to Discovery Goals
2. Keyword Density
3. Entity Mentions
4. Temporal Relevance
5. Document Credibility
6. Information Uniqueness

For each sub-score, provide a brief justification.

Output your analysis in the following format:
RELEVANCE_SCORE: [0-100]
RELEVANCE_JUSTIFICATION: [Brief explanation]

KEYWORD_SCORE: [0-100]
KEYWORD_JUSTIFICATION: [Brief explanation]

ENTITY_SCORE: [0-100]
ENTITY_JUSTIFICATION: [Brief explanation]

TEMPORAL_SCORE: [0-100]
TEMPORAL_JUSTIFICATION: [Brief explanation]

CREDIBILITY_SCORE: [0-100]
CREDIBILITY_JUSTIFICATION: [Brief explanation]

UNIQUENESS_SCORE: [0-100]
UNIQUENESS_JUSTIFICATION: [Brief explanation]
"""

# Dossier Section Compilation
dossier_section_prompt = """
Compile all the information gathered about this document into a cohesive dossier section.

Document Information:
{doc_info}

Relevance Analysis:
{relevance_analysis}

Key Extracts:
{key_extracts}

Tags:
{tags}

Explanation:
{explanation}

Importance Score:
{importance_score}

Discovery Goals:
{discovery_goals}

Keywords Found:
{keywords_found}

Entities Mentioned:
{entities_mentioned}

Compile this information into a well-formatted markdown section for the dossier. Include all relevant information and ensure it's presented in a clear, professional manner. The section should include:

1. Document summary (type, date, author, recipient, subject)
2. Relevance to specific discovery goals
3. Key extracts with explanations of their relevance
4. Tags
5. List of keywords and entities found
6. Comprehensive explanation of importance
7. Importance score with justification

Format the section to be easily readable and scannable, using appropriate markdown headers, bullet points, and emphasis where needed.

[Your compiled markdown section here]
"""

# GPU Check (for local LLM model inference)
def is_gpu_available():
    if not GPU_AVAILABLE:
        logging.warning("GPU support not available: nvgpu module not found")
        return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0, "error": "nvgpu module not found"}
    try:
        gpu_info = nvgpu.gpu_info()
        num_gpus = len(gpu_info)
        if num_gpus == 0:
            logging.warning("No GPUs found on the system")
            return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0}
        first_gpu_vram = gpu_info[0]['mem_total']
        total_vram = sum(gpu['mem_total'] for gpu in gpu_info)
        logging.info(f"GPU(s) found: {num_gpus}, Total VRAM: {total_vram} MB")
        return {"gpu_found": True, "num_gpus": num_gpus, "first_gpu_vram": first_gpu_vram, "total_vram": total_vram, "gpu_info": gpu_info}
    except Exception as e:
        logging.error(f"Error checking GPU availability: {e}")
        return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0, "error": str(e)}

# Local LLM Model Download
async def download_models() -> Tuple[List[str], List[Dict[str, str]]]:
    download_status = []    
    model_url = "https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-GGUF/resolve/main/Llama-3.1-8B-Lexi-Uncensored_Q5_fixedrope.gguf"
    model_name = os.path.basename(model_url)
    current_file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(current_file_path)
    models_dir = os.path.join(base_dir, 'models')
    
    os.makedirs(models_dir, exist_ok=True)
    lock = FileLock(os.path.join(models_dir, "download.lock"))
    status = {"url": model_url, "status": "success", "message": "File already exists."}
    filename = os.path.join(models_dir, model_name)
    
    try:
        with lock.acquire(timeout=1200):
            if not os.path.exists(filename):
                logging.info(f"Downloading model {model_name} from {model_url}...")
                urllib.request.urlretrieve(model_url, filename)
                file_size = os.path.getsize(filename) / (1024 * 1024)
                if file_size < 100:
                    os.remove(filename)
                    status["status"] = "failure"
                    status["message"] = f"Downloaded file is too small ({file_size:.2f} MB), probably not a valid model file."
                    logging.error(f"Error: {status['message']}")
                else:
                    logging.info(f"Successfully downloaded: {filename} (Size: {file_size:.2f} MB)")
            else:
                logging.info(f"Model file already exists: {filename}")
    except Timeout:
        logging.error(f"Error: Could not acquire lock for downloading {model_name}")
        status["status"] = "failure"
        status["message"] = "Could not acquire lock for downloading."
    
    download_status.append(status)
    logging.info("Model download process completed.")
    return [model_name], download_status

# Local LLM Model Loading
def load_model(llm_model_name: str, raise_exception: bool = True):
    global USE_VERBOSE
    try:
        current_file_path = os.path.abspath(__file__)
        base_dir = os.path.dirname(current_file_path)
        models_dir = os.path.join(base_dir, 'models')
        matching_files = glob.glob(os.path.join(models_dir, f"{llm_model_name}*"))
        if not matching_files:
            logging.error(f"Error: No model file found matching: {llm_model_name}")
            raise FileNotFoundError
        model_file_path = max(matching_files, key=os.path.getmtime)
        logging.info(f"Loading model: {model_file_path}")
        try:
            logging.info("Attempting to load model with GPU acceleration...")
            model_instance = Llama(
                model_path=model_file_path,
                n_ctx=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS,
                verbose=USE_VERBOSE,
                n_gpu_layers=-1
            )
            logging.info("Model loaded successfully with GPU acceleration.")
        except Exception as gpu_e:
            logging.warning(f"Failed to load model with GPU acceleration: {gpu_e}")
            logging.info("Falling back to CPU...")
            try:
                model_instance = Llama(
                    model_path=model_file_path,
                    n_ctx=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS,
                    verbose=USE_VERBOSE,
                    n_gpu_layers=0
                )
                logging.info("Model loaded successfully with CPU.")
            except Exception as cpu_e:
                logging.error(f"Failed to load model with CPU: {cpu_e}")
                if raise_exception:
                    raise
                return None
        return model_instance
    except Exception as e:
        logging.error(f"Exception occurred while loading the model: {e}")
        traceback.print_exc()
        if raise_exception:
            raise
        return None

# API Interaction Functions
@backoff.on_exception(backoff.expo, 
                        (RateLimitError, APIError),
                        max_tries=5)
async def api_request_with_retry(client, *args, **kwargs):
    try:
        return await client(*args, **kwargs)
    except Exception as e:
        logging.error(f"API request failed after multiple retries: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def retry_api_call(func, *args, **kwargs):
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logging.error(f"API call failed: {str(e)}")
        raise
    
async def generate_completion(prompt: str, max_tokens: int = 5000) -> Optional[str]:
    if USE_LOCAL_LLM:
        return await generate_completion_from_local_llm(DEFAULT_LOCAL_MODEL_NAME, prompt, max_tokens)
    elif API_PROVIDER == "CLAUDE":
        safe_max_tokens = calculate_safe_max_tokens(len(prompt), CLAUDE_MAX_TOKENS)
        return await retry_api_call(generate_completion_from_claude, prompt, safe_max_tokens)
    elif API_PROVIDER == "OPENAI":
        safe_max_tokens = calculate_safe_max_tokens(len(prompt), OPENAI_MAX_TOKENS)
        return await retry_api_call(generate_completion_from_openai, prompt, safe_max_tokens)
    else:
        logging.error(f"Invalid API_PROVIDER: {API_PROVIDER}")
        return None

def get_tokenizer(model_name: str):
    if model_name.startswith("gpt-4o"):
        return tiktoken.encoding_for_model('gpt-4o')
    elif model_name.startswith("gpt-"):
        return tiktoken.encoding_for_model(model_name)
    elif model_name.startswith("claude-"):
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", clean_up_tokenization_spaces=False)
    elif model_name.startswith("llama-"):
        return AutoTokenizer.from_pretrained("huggyllama/llama-7b", clean_up_tokenization_spaces=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def estimate_tokens(text: str, model_name: str) -> int:
    try:
        tokenizer = get_tokenizer(model_name)
        return len(tokenizer.encode(text))
    except Exception as e:
        logging.warning(f"Error using tokenizer for {model_name}: {e}. Falling back to approximation.")
        return approximate_tokens(text)

def approximate_tokens(text: str) -> int:
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Split on whitespace and punctuation, keeping punctuation
    tokens = re.findall(r'\b\w+\b|\S', text)
    count = 0
    for token in tokens:
        if token.isdigit():
            count += max(1, len(token) // 2)  # Numbers often tokenize to multiple tokens
        elif re.match(r'^[A-Z]{2,}$', token):  # Acronyms
            count += len(token)
        elif re.search(r'[^\w\s]', token):  # Punctuation and special characters
            count += 1
        elif len(token) > 10:  # Long words often split into multiple tokens
            count += len(token) // 4 + 1
        else:
            count += 1
    # Add a 10% buffer for potential underestimation
    return int(count * 1.1)

def chunk_text(text: str, max_chunk_tokens: int, model_name: str) -> List[str]:
    chunks = []
    tokenizer = get_tokenizer(model_name)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = []
    current_chunk_tokens = 0
    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence))
        if sentence_tokens > max_chunk_tokens:
            # If a single sentence is too long, split it into smaller parts
            sentence_parts = split_long_sentence(sentence, max_chunk_tokens, model_name)
            for part in sentence_parts:
                part_tokens = len(tokenizer.encode(part))
                if current_chunk_tokens + part_tokens > max_chunk_tokens:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [part]
                    current_chunk_tokens = part_tokens
                else:
                    current_chunk.append(part)
                    current_chunk_tokens += part_tokens
        elif current_chunk_tokens + sentence_tokens > max_chunk_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_chunk_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_tokens
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def split_long_sentence(sentence: str, max_tokens: int, model_name: str) -> List[str]:
    words = sentence.split()
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    tokenizer = get_tokenizer(model_name)
    for word in words:
        word_tokens = len(tokenizer.encode(word))
        if current_chunk_tokens + word_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_chunk_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_chunk_tokens += word_tokens
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def adjust_overlaps(chunks: List[str], tokenizer, max_chunk_tokens: int, overlap_size: int = 50) -> List[str]:
    adjusted_chunks = []
    for i in range(len(chunks)):
        if i == 0:
            adjusted_chunks.append(chunks[i])
        else:
            overlap_tokens = len(tokenizer.encode(' '.join(chunks[i-1].split()[-overlap_size:])))
            current_tokens = len(tokenizer.encode(chunks[i]))
            if overlap_tokens + current_tokens > max_chunk_tokens:
                overlap_adjusted = chunks[i].split()[:-overlap_size]
                adjusted_chunks.append(' '.join(overlap_adjusted))
            else:
                adjusted_chunks.append(' '.join(chunks[i-1].split()[-overlap_size:] + chunks[i].split()))
    return adjusted_chunks

async def generate_completion_from_claude(prompt: str, max_tokens: int = CLAUDE_MAX_TOKENS - TOKEN_BUFFER) -> Optional[str]:
    if not ANTHROPIC_API_KEY:
        logging.error("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
        return None
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    prompt_tokens = estimate_tokens(prompt, CLAUDE_MODEL_STRING)
    adjusted_max_tokens = min(max_tokens, CLAUDE_MAX_TOKENS - prompt_tokens - TOKEN_BUFFER)
    if adjusted_max_tokens <= 0:
        logging.warning("Prompt is too long for Claude API. Chunking the input.")
        chunks = chunk_text(prompt, CLAUDE_MAX_TOKENS - TOKEN_CUSHION, CLAUDE_MODEL_STRING)
        results = []
        for chunk in chunks:
            try:
                async with client.messages.stream(
                    model=CLAUDE_MODEL_STRING,
                    max_tokens=CLAUDE_MAX_TOKENS // 2,
                    temperature=0.7,
                    messages=[{"role": "user", "content": chunk}],
                ) as stream:
                    message = await stream.get_final_message()
                    results.append(message.content[0].text)
            except Exception as e:
                logging.error(f"An error occurred while processing a chunk: {e}")
        return " ".join(results)
    else:
        try:
            async with client.messages.stream(
                model=CLAUDE_MODEL_STRING,
                max_tokens=adjusted_max_tokens,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = await stream.get_final_message()
                output_text = message.content[0].text
                logging.info(f"Total input tokens: {message.usage.input_tokens:,}")
                logging.info(f"Total output tokens: {message.usage.output_tokens:,}")
                logging.info(f"Generated output (abbreviated): {' '.join(output_text[:150].replace('\\r', '').replace('\\n', '').split())}...")
                return output_text
        except Exception as e:
            logging.error(f"An error occurred while requesting from Claude API: {e}")
            return None

async def generate_completion_from_openai(prompt: str, max_tokens: int = 16384) -> Optional[str]:
    if not OPENAI_API_KEY:
        logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None
    prompt_tokens = estimate_tokens(prompt, OPENAI_COMPLETION_MODEL)
    adjusted_max_tokens = min(max_tokens, OPENAI_MAX_TOKENS - prompt_tokens - TOKEN_BUFFER)
    use_verbose_openai_logging = 0
    if use_verbose_openai_logging:
        logging.info(f"Prompt tokens: {prompt_tokens:,}")
        logging.info(f"Max tokens allowed: {adjusted_max_tokens:,}")
        logging.info(f"Total context length: {prompt_tokens + adjusted_max_tokens:,}")
    if adjusted_max_tokens <= 1:
        logging.warning(f"Prompt is too long for OpenAI API. Prompt tokens: {prompt_tokens:,}")
        return prompt  # Return the original prompt if it's too long
    try:
        response = await api_request_with_retry(
            openai_client.chat.completions.create,
            model=OPENAI_COMPLETION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=adjusted_max_tokens,
            temperature=0.7,
        )
        output_text = response.choices[0].message.content
        if use_verbose_openai_logging:
            logging.info(f"Total tokens: {response.usage.total_tokens:,}")
            logging.info(f"Generated output (abbreviated): {' '.join(output_text[:150].replace('\\r', '').replace('\\n', '').split())}...")
        return output_text
    except (RateLimitError, APIError) as e:
        logging.error(f"OpenAI API error: {str(e)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while requesting from OpenAI API: {str(e)}")
    return prompt  # Return the original prompt if any error occurs
    
async def generate_completion_from_local_llm(llm_model_name: str, input_prompt: str, number_of_tokens_to_generate: int = 100, temperature: float = 0.7, grammar_file_string: str = None):
    logging.info(f"Starting text completion using model: '{llm_model_name}' for input prompt: '{input_prompt}'")
    llm = load_model(llm_model_name)
    prompt_tokens = estimate_tokens(input_prompt, llm_model_name)
    adjusted_max_tokens = min(number_of_tokens_to_generate, LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - prompt_tokens - TOKEN_BUFFER)
    if adjusted_max_tokens <= 0:
        logging.warning("Prompt is too long for LLM. Chunking the input.")
        chunks = chunk_text(input_prompt, LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - TOKEN_CUSHION, llm_model_name)
        results = []
        for chunk in chunks:
            try:
                output = llm(
                    prompt=chunk,
                    max_tokens=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - TOKEN_CUSHION,
                    temperature=temperature,
                )
                results.append(output['choices'][0]['text'])
            except Exception as e:
                logging.error(f"An error occurred while processing a chunk: {e}")
        return " ".join(results)
    else:
        grammar_file_string_lower = grammar_file_string.lower() if grammar_file_string else ""
        if grammar_file_string_lower:
            list_of_grammar_files = glob.glob("./grammar_files/*.gbnf")
            matching_grammar_files = [x for x in list_of_grammar_files if grammar_file_string_lower in os.path.splitext(os.path.basename(x).lower())[0]]
            if len(matching_grammar_files) == 0:
                logging.error(f"No grammar file found matching: {grammar_file_string}")
                raise FileNotFoundError
            grammar_file_path = max(matching_grammar_files, key=os.path.getmtime)
            logging.info(f"Loading selected grammar file: '{grammar_file_path}'")
            llama_grammar = LlamaGrammar.from_file(grammar_file_path)
            output = llm(
                prompt=input_prompt,
                max_tokens=adjusted_max_tokens,
                temperature=temperature,
                grammar=llama_grammar
            )
        else:
            output = llm(
                prompt=input_prompt,
                max_tokens=adjusted_max_tokens,
                temperature=temperature
            )
        generated_text = output['choices'][0]['text']
        if grammar_file_string == 'json':
            generated_text = generated_text.encode('unicode_escape').decode()
        finish_reason = str(output['choices'][0]['finish_reason'])
        llm_model_usage_json = json.dumps(output['usage'])
        logging.info(f"Completed text completion in {output['usage']['total_time']:.2f} seconds. Beginning of generated text: \n'{generated_text[:150]}'...")
        return {
            "generated_text": generated_text,
            "finish_reason": finish_reason,
            "llm_model_usage_json": llm_model_usage_json
        }

def calculate_safe_max_tokens(input_length: int, model_max_tokens: int = 128000, token_buffer: int = 1000) -> int:
    available_tokens = max(0, model_max_tokens - input_length - token_buffer)
    return min(available_tokens, 16384)  # Cap at max output tokens

def calculate_importance_score(importance_analysis: str) -> dict:
    # Extract sub-scores and justifications
    sub_scores = {}
    pattern = r'(\w+)_SCORE: (\d+)\n(\w+)_JUSTIFICATION: (.+)'
    matches = re.findall(pattern, importance_analysis, re.MULTILINE)
    for match in matches:
        category = match[0].lower()
        score = int(match[1])
        justification = match[3].strip()
        sub_scores[category] = {"score": score, "justification": justification}
    # Define weights for each category
    weights = {
        "relevance": 0.3,
        "keyword": 0.2,
        "entity": 0.2,
        "temporal": 0.1,
        "credibility": 0.1,
        "uniqueness": 0.1
    }
    # Calculate weighted average
    total_score = sum(sub_scores[category]["score"] * weights[category] for category in weights)
    # Round to two decimal places
    final_score = round(total_score, 2)
    # Prepare detailed breakdown
    breakdown = {
        "final_score": final_score,
        "sub_scores": sub_scores,
        "explanation": f"The final importance score of {final_score} is a weighted average of the sub-scores, "
                        f"with the following weights: {weights}"
    }
    return breakdown

async def process_chunk_multi_stage(chunk, chunk_index, total_chunks, discovery_params):
    api_calls = 0
    total_tokens = 0
    model_name = OPENAI_COMPLETION_MODEL if API_PROVIDER == "OPENAI" else CLAUDE_MODEL_STRING

    logging.info(f"Starting processing of chunk {chunk_index + 1}/{total_chunks}")

    async def call_api(prompt, max_tokens, stage_name):
        nonlocal api_calls, total_tokens
        api_calls += 1
        logging.info(f"Chunk {chunk_index + 1}/{total_chunks}: Calling API for {stage_name} (API call #{api_calls})")
        response = await generate_completion(prompt, max_tokens)
        tokens_used = estimate_tokens(prompt, model_name) + estimate_tokens(response, model_name)
        total_tokens += tokens_used
        logging.info(f"Chunk {chunk_index + 1}/{total_chunks}: Completed {stage_name}. Tokens used: {tokens_used}")
        return response

    try:
        # Pre-calculate token counts for prompt templates
        doc_id_prompt_tokens = estimate_tokens(doc_id_prompt, model_name)
        relevance_check_prompt_tokens = estimate_tokens(relevance_check_prompt, model_name)
        extract_gen_prompt_tokens = estimate_tokens(extract_gen_prompt, model_name)
        tag_gen_prompt_tokens = estimate_tokens(tag_gen_prompt, model_name)
        explanation_gen_prompt_tokens = estimate_tokens(explanation_gen_prompt, model_name)
        importance_score_prompt_tokens = estimate_tokens(importance_score_prompt, model_name)

        # Calculate available tokens for each stage
        available_tokens = OPENAI_MAX_TOKENS - TOKEN_BUFFER
        doc_id_tokens = min(500, available_tokens - doc_id_prompt_tokens)
        relevance_tokens = min(500, available_tokens - relevance_check_prompt_tokens)
        extract_tokens = min(1000, available_tokens - extract_gen_prompt_tokens)
        tag_tokens = min(500, available_tokens - tag_gen_prompt_tokens)
        explanation_tokens = min(1000, available_tokens - explanation_gen_prompt_tokens)
        importance_tokens = min(1000, available_tokens - importance_score_prompt_tokens)

        logging.info(f"Chunk {chunk_index + 1}/{total_chunks}: Token allocations - Doc ID: {doc_id_tokens}, Relevance: {relevance_tokens}, Extract: {extract_tokens}, Tag: {tag_tokens}, Explanation: {explanation_tokens}, Importance: {importance_tokens}")

        # Stage 1: Document Identification
        doc_info = await call_api(doc_id_prompt.format(document_excerpt=chunk, entities_of_interest=discovery_params['entities_of_interest']), doc_id_tokens, "Document Identification")

        # Stage 2: Relevance Check
        relevance_analysis = await call_api(relevance_check_prompt.format(
            discovery_goals=discovery_params['discovery_goals'],
            doc_info=doc_info,
            document_excerpt=chunk,
            keywords=discovery_params['keywords'],
            entities_of_interest=discovery_params['entities_of_interest']
        ), relevance_tokens, "Relevance Check")

        if 'RELEVANT: No' in relevance_analysis:
            logging.info(f"Chunk {chunk_index + 1}/{total_chunks}: Deemed not relevant. Skipping further processing.")
            return None, api_calls, total_tokens

        logging.info(f"Chunk {chunk_index + 1}/{total_chunks}: Deemed relevant. Proceeding with further analysis.")

        # Parallel processing stages
        logging.info(f"Chunk {chunk_index + 1}/{total_chunks}: Starting parallel processing of Extract Generation and Tag Generation")
        extract_task = asyncio.create_task(call_api(extract_gen_prompt.format(
            relevance_analysis=relevance_analysis,
            full_document_text=chunk,
            discovery_goals=discovery_params['discovery_goals'],
            keywords=discovery_params['keywords'],
            entities_of_interest=discovery_params['entities_of_interest']
        ), extract_tokens, "Extract Generation"))

        tag_task = asyncio.create_task(call_api(tag_gen_prompt.format(
            doc_info=doc_info,
            relevance_analysis=relevance_analysis,
            discovery_goals=discovery_params['discovery_goals'],
            keywords=discovery_params['keywords'],
            entities_of_interest=discovery_params['entities_of_interest']
        ), tag_tokens, "Tag Generation"))

        key_extracts, tags = await asyncio.gather(extract_task, tag_task)
        logging.info(f"Chunk {chunk_index + 1}/{total_chunks}: Completed parallel processing of Extract Generation and Tag Generation")

        # Sequential stages
        explanation = await call_api(explanation_gen_prompt.format(
            discovery_goals=discovery_params['discovery_goals'],
            doc_info=doc_info,
            key_extracts=key_extracts,
            relevance_analysis=relevance_analysis,
            keywords=discovery_params['keywords'],
            entities_of_interest=discovery_params['entities_of_interest']
        ), explanation_tokens, "Explanation Generation")

        importance_analysis = await call_api(importance_score_prompt.format(
            doc_info=doc_info,
            relevance_analysis=relevance_analysis,
            key_extracts=key_extracts,
            tags=tags,
            explanation=explanation,
            discovery_goals=discovery_params['discovery_goals'],
            keywords_found=discovery_params['keywords'],
            entities_mentioned=discovery_params['entities_of_interest']
        ), importance_tokens, "Importance Score Analysis")

        importance_score = calculate_importance_score(importance_analysis)
        logging.info(f"Chunk {chunk_index + 1}/{total_chunks}: Calculated importance score: {importance_score['final_score']:.2f}")

        chunk_package = {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "doc_info": doc_info,
            "relevance": relevance_analysis,
            "key_extracts": key_extracts,
            "tags": tags,
            "explanation": explanation,
            "importance_score": importance_score,
            "api_calls": api_calls,
            "total_tokens": total_tokens,
            "chunk_text": chunk
        }
        logging.info(f"Chunk {chunk_index + 1}/{total_chunks}: Processing complete. API calls: {api_calls}, Total tokens used: {total_tokens}")
        return chunk_package, api_calls, total_tokens
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_index + 1}/{total_chunks}: {str(e)}")
        return None, api_calls, total_tokens

def compile_dossier_section(processed_chunks: List[Dict[str, Any]], document_id: str, file_path: str, discovery_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine all processed chunks into a single dossier section.
    This function intelligently merges the information from all chunks
    and creates a cohesive section for the entire document.
    """
    if not processed_chunks:
        return None
    # Initialize the merged data
    merged_data = {
        "document_id": document_id,
        "file_path": file_path,
        "doc_info": processed_chunks[0]["doc_info"],
        "relevance": [],
        "key_extracts": [],
        "tags": set(),
        "explanations": set(),
        "importance_scores": []
    }
    # Merge data from all chunks
    for chunk in processed_chunks:
        merged_data["relevance"].extend(chunk["relevance"].split(", "))
        merged_data["key_extracts"].extend(chunk["key_extracts"])
        merged_data["tags"].update(chunk["tags"].split(", "))
        merged_data["explanations"].add(chunk["explanation"])
        merged_data["importance_scores"].append(chunk["importance_score"])
    # Process and deduplicate relevance
    relevance_counter = Counter(merged_data["relevance"])
    merged_data["relevance"] = [{"goal": goal, "frequency": count} for goal, count in relevance_counter.items()]
    merged_data["relevance"].sort(key=lambda x: x["frequency"], reverse=True)
    # Deduplicate and sort key extracts
    merged_data["key_extracts"] = sorted(set(merged_data["key_extracts"]), key=lambda x: merged_data["key_extracts"].index(x))
    # Sort tags by frequency
    tag_counter = Counter(merged_data["tags"])
    merged_data["tags"] = [{"tag": tag, "frequency": count} for tag, count in tag_counter.most_common()]
    # Combine explanations
    merged_data["explanation"] = " ".join(merged_data["explanations"])
    # Calculate the overall importance score
    overall_importance = sum(score["final_score"] for score in merged_data["importance_scores"]) / len(merged_data["importance_scores"])
    # Aggregate sub-scores
    aggregated_sub_scores = {
        category: {
            "score": sum(score["sub_scores"][category]["score"] for score in merged_data["importance_scores"]) / len(merged_data["importance_scores"]),
            "justifications": [score["sub_scores"][category]["justification"] for score in merged_data["importance_scores"]]
        }
        for category in merged_data["importance_scores"][0]["sub_scores"].keys()
    }
    # Generate a concise summary
    summary = generate_summary(merged_data, discovery_params, overall_importance)
    # Compile the final dossier section
    dossier_section = f"""## Document: {merged_data['doc_info']['TYPE']} - {merged_data['doc_info']['SUBJECT']}

**Document ID:** {document_id}
**File Path:** {file_path}

### Summary
{summary}

### Document Information
- **Type:** {merged_data['doc_info']['TYPE']}
- **Date:** {merged_data['doc_info']['DATE']}
- **Author:** {merged_data['doc_info']['AUTHOR']}
- **Recipient:** {merged_data['doc_info']['RECIPIENT']}
- **Subject:** {merged_data['doc_info']['SUBJECT']}

### Relevance to Discovery Goals
{format_relevance(merged_data['relevance'], discovery_params)}

### Key Extracts
{format_key_extracts(merged_data['key_extracts'], discovery_params['keywords'])}

### Tags
{format_tags(merged_data['tags'])}

### Detailed Explanation
{merged_data['explanation']}

### Importance Score: {overall_importance:.2f}
{format_importance_breakdown(aggregated_sub_scores, overall_importance)}

### Metadata
- **Document ID:** {document_id}
- **File Path:** {file_path}
- **Chunks Processed:** {len(processed_chunks)}
"""
    return {
        "document_id": document_id,
        "file_path": file_path,
        "dossier_section": dossier_section,
        "importance_score": overall_importance,
        "raw_data": merged_data
    }

def generate_summary(merged_data: Dict[str, Any], discovery_params: Dict[str, Any], overall_importance: float) -> str:
    """Generate a concise summary of the document's relevance and importance."""
    top_goals = ", ".join([goal["goal"] for goal in merged_data["relevance"][:3]])
    top_tags = ", ".join([tag["tag"] for tag in merged_data["tags"][:5]])
    summary = f"This {merged_data['doc_info']['TYPE'].lower()} is primarily relevant to the following discovery goals: {top_goals}. "
    summary += f"Key topics include: {top_tags}. "
    summary += f"The document's overall importance score is {overall_importance:.2f} out of 100, "
    summary += f"indicating {'high' if overall_importance > 75 else 'moderate' if overall_importance > 50 else 'low'} relevance to the case. "
    summary += f"Created on {merged_data['doc_info']['DATE']}, "
    summary += f"this document involves communication between {merged_data['doc_info']['AUTHOR']} and {merged_data['doc_info']['RECIPIENT']}."
    return summary

def format_relevance(relevance: List[Dict[str, Any]], discovery_params: Dict[str, Any]) -> str:
    """Format the relevance information, including goal descriptions from discovery_params."""
    formatted_relevance = []
    for rel in relevance:
        goal_description = next((goal["description"] for goal in discovery_params["discovery_goals"] if goal["description"].startswith(rel["goal"])), "Unknown goal")
        formatted_relevance.append(f"- **{rel['goal']}** (Frequency: {rel['frequency']})\n  {goal_description}")
    return "\n".join(formatted_relevance)

def format_key_extracts(extracts: List[str], keywords: List[str], max_extracts: int = 5) -> str:
    """Format and limit the number of key extracts, highlighting keywords and critical parts."""
    formatted_extracts = []
    for i, extract in enumerate(extracts[:max_extracts], 1):
        # Highlight keywords in bold
        highlighted_extract = highlight_keywords(extract, keywords)
        # Add emphasis to critical parts (assuming they are marked with asterisks)
        emphasized_extract = re.sub(r'\*\*(.*?)\*\*', r'***\1***', highlighted_extract)
        # Add ellipses for omitted parts (assuming they are marked with ...)
        final_extract = re.sub(r'\s*\.\.\.\s*', ' ... ', emphasized_extract)
        formatted_extracts.append(f"{i}. {final_extract}")
    return "\n".join(formatted_extracts)

def format_tags(tags: List[Dict[str, Any]], max_tags: int = 10) -> str:
    """Format tags with their frequencies."""
    return ", ".join([f"**{tag['tag']}** ({tag['frequency']})" for tag in tags[:max_tags]])

def highlight_keywords(text: str, keywords: List[str]) -> str:
    """Highlight keywords in the text using bold markdown."""
    for keyword in keywords:
        text = re.sub(rf'\b{re.escape(keyword)}\b', f'**{keyword}**', text, flags=re.IGNORECASE)
    return text

def format_importance_breakdown(sub_scores: Dict[str, Dict[str, Any]], overall_importance: float) -> str:
    """Format the importance score breakdown."""
    breakdown = "#### Importance Score Breakdown\n"
    for category, details in sub_scores.items():
        breakdown += f"- **{category.capitalize()}**: {details['score']:.2f}\n"
        breakdown += f"  - Justification: {'; '.join(set(details['justifications']))}\n"
    breakdown += f"\nThe overall importance score of {overall_importance:.2f} is a weighted average of these sub-scores."
    return breakdown

async def generate_discovery_config(user_input: str) -> str:
    async def call_llm(prompt: str) -> str:
        return await generate_completion(prompt)
    # Step 1: Generate JSON configuration from user input
    json_generation_prompt = f"""
    Based on the following user input about their discovery goals and case summary, generate a comprehensive JSON configuration file for legal discovery automation. The JSON should include the following fields:

    - case_name: A string representing the name of the case
    - discovery_goals: An array of objects, each containing:
      - description: A string describing the discovery goal
      - keywords: An array of relevant keywords (derived from the user input)
      - importance: A number from 1 to 10 indicating the importance of this goal (derived from the user input)
    - entities_of_interest: An array of strings representing relevant entities (people, companies, etc.)
    - minimum_importance_score: A number from 0.0 to 100.0 representing the minimum score for a document to be included in the final dossier

    Make sure all the information from the user input is accurately represented, and add any missing details where appropriate.

    User input:
    {user_input}

    Generate the JSON configuration:
    """
    json_config = await call_llm(json_generation_prompt)
    # Step 2: Validate and fix the JSON structure with retry logic
    validation_prompt_template = """
    The following is a JSON configuration for legal discovery automation. Please verify that it follows the required structure and fix any issues:

    {json_config}

    The JSON should have the following structure:
    {{
      "case_name": string,
      "discovery_goals": [
        {{
          "description": string,
          "keywords": [string],
          "importance": number (1-10)
        }}
      ],
      "entities_of_interest": [string],
      "minimum_importance_score": number (0.0-100.0)
    }}

    Ensure that all discovery goals have corresponding keywords and importance levels, and that the entities of interest are accurately captured.

    Provide the corrected JSON below; IMPORTANT: DO NOT REMOVE OR ABBREVIATE ANY CONTENT FROM THE JSON DATA!!! AND ONLY RESPOND WITH THE CORRECTED JSON, DO NOT ADD ANY EXTRA INFORMATION OR ANY OTHER TEXT, NOT EVEN MARKDOWN CODE BLOCKS:
    """
    MAX_RETRIES = 5
    retries = 0
    while retries < MAX_RETRIES:
        try:
            corrected_json = await call_llm(validation_prompt_template.format(json_config=json_config))
            config_dict = json.loads(corrected_json)
            validate_config_structure(config_dict)
            break  # If successful, exit the loop
        except json.JSONDecodeError:
            logging.error(f"Attempt {retries + 1}: Failed to parse the JSON. Retrying...")
            retries += 1
        except ValueError as e:
            logging.error(f"Attempt {retries + 1}: Invalid configuration structure: {str(e)}. Retrying...")
            retries += 1
        json_config = corrected_json  # Update with the latest version from the LLM
    else:
        raise ValueError("Exceeded maximum retries to generate a valid JSON configuration.")
    # Step 3: Generate a descriptive file name
    file_name_prompt = f"""
    Based on the following case information, generate a short, descriptive file name for the JSON configuration file. Use only lowercase letters and underscores, and end the file name with '.json'.

    Case information:
    {json.dumps(config_dict, indent=2)}

    Generate file name (ONLY RESPOND WITH THE CORRECTED FILE NAME, DO NOT ADD ANY EXTRA INFORMATION OR ANY OTHER TEXT, NOT EVEN MARKDOWN CODE BLOCKS):
    """
    file_name = await call_llm(file_name_prompt)
    file_name = re.sub(r'[^a-z_.]', '', file_name.strip().lower())
    if not file_name.endswith('.json'):
        file_name += '.json'
    # Step 4: Save the JSON file
    config_folder = 'discovery_configuration_json_files'
    os.makedirs(config_folder, exist_ok=True)
    file_path = os.path.join(config_folder, file_name)
    with open(file_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logging.info(f"Configuration saved successfully to {file_path}")
    return file_path

def validate_config_structure(config: Dict[str, Any]):
    required_fields = {'case_name', 'discovery_goals', 'entities_of_interest', 'minimum_importance_score'}
    if not all(field in config for field in required_fields):
        raise ValueError("Missing required fields in the configuration.")
    if not isinstance(config['case_name'], str):
        raise ValueError("case_name must be a string.")
    if not isinstance(config['discovery_goals'], list):
        raise ValueError("discovery_goals must be a list.")
    for goal in config['discovery_goals']:
        if not all(field in goal for field in ['description', 'keywords', 'importance']):
            raise ValueError("Each discovery goal must have description, keywords, and importance.")
        if not isinstance(goal['description'], str):
            raise ValueError("Goal description must be a string.")
        if not isinstance(goal['keywords'], list) or not all(isinstance(k, str) for k in goal['keywords']):
            raise ValueError("Goal keywords must be a list of strings.")
        if not isinstance(goal['importance'], (int, float)) or not 1 <= goal['importance'] <= 10:
            raise ValueError("Goal importance must be a number between 1 and 10.")
    if not isinstance(config['entities_of_interest'], list) or not all(isinstance(e, str) for e in config['entities_of_interest']):
        raise ValueError("entities_of_interest must be a list of strings.")
    if not isinstance(config['minimum_importance_score'], (int, float)) or not 0 <= config['minimum_importance_score'] <= 100:
        raise ValueError("minimum_importance_score must be a number between 0 and 100.")

def compile_final_dossier(dossier_sections: List[Dict[str, Any]]) -> str:
    # Sort dossier sections by importance score in descending order
    sorted_sections = sorted(dossier_sections, key=lambda x: x['importance_score'], reverse=True)
    # Compile the final dossier
    final_dossier = "# Legal Discovery Dossier\n\n"
    final_dossier += f"Total Documents Processed: {len(sorted_sections)}\n\n"
    final_dossier += "## Table of Contents\n\n"
    for i, section in enumerate(sorted_sections, 1):
        doc_title = f"{section['doc_info']['TYPE']} - {section['doc_info']['SUBJECT']}"
        final_dossier += f"{i}. [{doc_title}](#{i}-{doc_title.lower().replace(' ', '-')})\n"
    final_dossier += "\n## Document Summaries\n\n"
    for i, section in enumerate(sorted_sections, 1):
        final_dossier += f"### {i}. {section['doc_info']['TYPE']} - {section['doc_info']['SUBJECT']}\n\n"
        final_dossier += section['dossier_section']
        final_dossier += "\n---\n\n"
    return final_dossier
        
async def process_pst_file(file_path: str, converted_source_dir: str) -> int:
    MAX_EMAILS_PER_BUNDLE = 1000  # Maximum number of emails per bundle
    semaphore = asyncio.Semaphore(MAX_EMAILS_PER_BUNDLE)  # Limit concurrency to avoid overloading the system
    logging.info(f"Processing PST file: {file_path}")
    try:
        pst = pypff.file()
        pst.open(file_path)
        root = pst.get_root_folder()
        emails_by_sender = defaultdict(list)
        tasks = []
        async def process_message(message):
            async with semaphore:
                sender = message.get_sender_name()
                sender_email = message.get_sender_email_address()
                if not sender_email:
                    _, sender_email = parseaddr(sender)
                subject = message.get_subject()
                body = message.get_plain_text_body()
                date = message.get_delivery_time()
                if date:
                    date = parsedate_to_datetime(date).isoformat()
                recipients = message.get_recipients()
                to_list, cc_list, bcc_list = [], [], []
                for recipient in recipients:
                    name, email = recipient.get_name(), recipient.get_email()
                    recipient_type = recipient.get_type()
                    if recipient_type == pypff.message_recipient_types.TO:
                        to_list.append({"name": name, "email": email})
                    elif recipient_type == pypff.message_recipient_types.CC:
                        cc_list.append({"name": name, "email": email})
                    elif recipient_type == pypff.message_recipient_types.BCC:
                        bcc_list.append({"name": name, "email": email})
                headers_str = json.dumps({
                    "From": {"name": sender, "email": sender_email},
                    "To": to_list,
                    "Cc": cc_list,
                    "Bcc": bcc_list,
                    "Subject": subject,
                    "Date": date
                }, sort_keys=True)
                content_for_hash = headers_str + (body or "")
                email_unique_identifier = hashlib.sha256(content_for_hash.encode()).hexdigest()[:16]
                return {
                    "unique_identifier": email_unique_identifier,
                    "sender": {"name": sender, "email": sender_email},
                    "to": to_list,
                    "cc": cc_list,
                    "bcc": bcc_list,
                    "subject": subject,
                    "date": date,
                    "body": body
                }
        async def process_folder(folder):
            folder_tasks = []
            for message in folder.sub_messages:
                folder_tasks.append(asyncio.create_task(process_message(message)))
            for sub_folder in folder.sub_folders:
                folder_tasks.extend(await process_folder(sub_folder))
            return folder_tasks
        tasks = await process_folder(root)
        email_data_list = await asyncio.gather(*tasks)
        for email_data in email_data_list:
            if email_data:
                emails_by_sender[email_data['sender']['email']].append(email_data)
        pst.close()
        async def write_email_bundle(sender_email, emails):
            safe_sender = ''.join(c.lower() if c.isalnum() else '_' for c in sender_email)
            markdown_file_name = f"email_bundle__sender__{safe_sender}__category__pst_file.md"
            markdown_file_path = os.path.join(converted_source_dir, markdown_file_name)
            markdown_content = ""
            for index, email in enumerate(emails[:MAX_EMAILS_PER_BUNDLE], 1):
                markdown_content += f"# Email {index} of {len(emails[:MAX_EMAILS_PER_BUNDLE])} from {email['sender']['name']} <{email['sender']['email']}>\n\n"
                markdown_content += f"**Unique Email Identifier:** {email['unique_identifier']}\n"
                markdown_content += f"**From:** {email['sender']['name']} <{email['sender']['email']}>\n"
                markdown_content += f"**To:** {', '.join([f'{r['name']} <{r['email']}>' for r in email['to']])}\n"
                if email['cc']:
                    markdown_content += f"**Cc:** {', '.join([f'{r['name']} <{r['email']}>' for r in email['cc']])}\n"
                if email['bcc']:
                    markdown_content += f"**Bcc:** {', '.join([f'{r['name']} <{r['email']}>' for r in email['bcc']])}\n"
                markdown_content += f"**Subject:** {email['subject']}\n"
                markdown_content += f"**Date:** {email['date']}\n\n"
                markdown_content += f"**Body:**\n\n{email['body']}\n\n"
                markdown_content += "---\n\n"
            async with aiofiles.open(markdown_file_path, 'w', encoding='utf-8') as f:
                await f.write(markdown_content)
            logging.info(f"Created markdown file: {markdown_file_name}")
        write_tasks = [write_email_bundle(sender_email, emails) for sender_email, emails in emails_by_sender.items()]
        await asyncio.gather(*write_tasks)
        logging.info(f"Successfully processed PST file: {file_path}")
        return len(emails_by_sender)  # Return the number of email bundles created
    except Exception as e:
        logging.error(f"Error processing PST file {file_path}: {str(e)}")
        raise

async def process_pst_files(original_source_dir: str, converted_source_dir: str):
    for filename in os.listdir(original_source_dir):
        if filename.lower().endswith('.pst'):
            file_path = os.path.join(original_source_dir, filename)
            logging.info(f"Processing MS Outlook PST file {file_path}")
            number_of_email_bundle_files_created = await process_pst_file(file_path, converted_source_dir)
            logging.info(f"Bundled {number_of_email_bundle_files_created} email bundles from {file_path}")
            
def process_iphone_messages(file_path: str) -> str:
    """
    Process iPhone messages database and bundle messages into markdown format.
    """
    logging.info(f"Processing iPhone messages database: {file_path}")
    try:
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                datetime(message.date/1000000000 + strftime("%s", "2001-01-01"), "unixepoch", "localtime") as date,
                message.text,
                handle.id as contact
            FROM 
                message 
                LEFT JOIN handle ON message.handle_id = handle.ROWID 
            ORDER BY 
                date
        """)
        messages = cursor.fetchall()
        conn.close()
        bundled_content = "# iPhone Messages\n\n"
        for date, text, contact in messages:
            bundled_content += f"**Date:** {date}\n"
            bundled_content += f"**Contact:** {contact}\n"
            bundled_content += f"**Message:** {text}\n\n---\n\n"
        logging.info(f"Successfully processed iPhone messages database: {file_path}")
        return bundled_content
    except Exception as e:
        logging.error(f"Error processing iPhone messages database {file_path}: {str(e)}")
        raise

def process_android_messages(file_path: str) -> str:
    """
    Process Android messages database and bundle messages into markdown format.
    """
    logging.info(f"Processing Android messages database: {file_path}")
    try:
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                datetime(date/1000, "unixepoch", "localtime") as date,
                body,
                address
            FROM 
                sms
            ORDER BY 
                date
        """)
        messages = cursor.fetchall()
        conn.close()
        bundled_content = "# Android Messages\n\n"
        for date, body, address in messages:
            bundled_content += f"**Date:** {date}\n"
            bundled_content += f"**Contact:** {address}\n"
            bundled_content += f"**Message:** {body}\n\n---\n\n"
        logging.info(f"Successfully processed Android messages database: {file_path}")
        return bundled_content
    except Exception as e:
        logging.error(f"Error processing Android messages database {file_path}: {str(e)}")
        raise

async def beautify_and_format_as_markdown(text: str) -> str:
    logging.info(f"Beautifying and formatting text as markdown (length: {len(text):,} characters); first 100 characters of source text: {text[:100]}")
    # Markdown formatting template
    markdown_template = """IMPORTANT: You are formatting a document for legal discovery. The integrity and accuracy of the original content are PARAMOUNT. Your task is ONLY to improve the formatting and readability using markdown, WITHOUT altering the original content in any way.

Reformat the following text as markdown, improving readability while preserving the approximate original structure and content EXACTLY. Follow these guidelines STRICTLY:

1. Convert existing headings to appropriate markdown heading levels (# for main titles, ## for subtitles, etc.)
2. Ensure each heading is on its own line with a blank line before and after
3. Maintain the EXACT original paragraph structure
4. Format existing lists properly (unordered or ordered) if they exist in the original text
5. Use emphasis (*italic*) and strong emphasis (**bold**) ONLY where they clearly exist in the original text
6. PRESERVE ALL ORIGINAL CONTENT AND MEANING EXACTLY - DO NOT ADD, REMOVE, OR CHANGE ANY WORDS
7. DO NOT add any extra punctuation or modify the existing punctuation
8. DO NOT add any introductory text, preamble, or markdown code block indicators
9. If tables are present, format them using markdown table syntax WITHOUT changing the content
10. For code snippets or technical content, use appropriate markdown code block formatting

WARNING: DO NOT GENERATE ANY NEW CONTENT. DO NOT SUMMARIZE. DO NOT EXPLAIN. ONLY FORMAT THE EXISTING TEXT.

CRITICAL: If you are unsure about any formatting decision, ALWAYS err on the side of preserving the original text exactly as it is.

Text to reformat:

{text}

Reformatted markdown (START DIRECTLY WITH THE CONTENT, NO PREAMBLE):
"""
    # Final filtering template
    filtering_template = """CRITICAL: You are reviewing a document formatted for legal discovery. The EXACT preservation of the original content is ESSENTIAL. Your task is ONLY to refine the markdown formatting, ensuring it adheres to best practices WITHOUT altering the original content in any way.

Review the following markdown-formatted text and refine it further. Follow these guidelines STRICTLY:

1. Ensure consistent heading levels and proper nesting WITHOUT changing the heading text
2. Verify that lists are properly formatted and indented WITHOUT altering list items
3. Check that emphasis and strong emphasis are used ONLY where clearly intended in the original
4. Ensure proper spacing between different elements (paragraphs, lists, headings) WITHOUT merging or splitting content
5. Verify that any code blocks or technical content are properly formatted WITHOUT changing the code
6. Ensure that tables, if present, are properly aligned and formatted WITHOUT altering table content
7. Remove any trailing whitespace or unnecessary blank lines
8. DO NOT remove any actual content or alter the meaning of the text in ANY WAY

WARNING: Your role is SOLELY to refine formatting. DO NOT add explanations, summaries, or any new content.

IMPORTANT: If there's any doubt about formatting, ALWAYS prioritize preserving the original text exactly as it is.

Text to refine:

{text}

Refined markdown (START DIRECTLY WITH THE CONTENT, NO PREAMBLE):
"""
    # Process the text with markdown formatting
    markdown_formatted = await process_text_with_llm(text, markdown_template, max_chunk_size=2000)
    # Apply final filtering
    refined_markdown = await process_text_with_llm(markdown_formatted, filtering_template, max_chunk_size=2000)
    logging.info(f"Text beautified and formatted as markdown. Output length: {len(refined_markdown):,} characters")
    return refined_markdown

async def preprocess_document(file_path: str, converted_source_dir: str) -> Tuple[str, str, Dict[str, Any], bool]:
    # Check if converted file already exists
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    txt_file_path = os.path.join(converted_source_dir, f"{base_name}.txt")
    md_file_path = os.path.join(converted_source_dir, f"{base_name}.md")
    if os.path.exists(txt_file_path) and os.path.getsize(txt_file_path) > 1024:
        logging.info(f"Using existing converted file: {txt_file_path}")
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            return file.read(), 'text/plain', {}, False
    elif os.path.exists(md_file_path) and os.path.getsize(md_file_path) > 1024:
        logging.info(f"Using existing converted file: {md_file_path}")
        with open(md_file_path, 'r', encoding='utf-8') as file:
            return file.read(), 'text/markdown', {}, True
    # If no existing converted file, proceed with conversion
    with open(file_path, 'rb') as file:
        file_content = file.read()
        result = magika.identify_bytes(file_content)
        detected_mime_type = result.output.mime_type
    logging.info(f"Detected MIME type for {file_path}: {detected_mime_type}")
    file_extension = os.path.splitext(file_path)[1].lower()
    metadata = {}
    used_smart_ocr = False
    already_in_markdown = False
    if file_path.lower().endswith(('.md', '.txt')) or detected_mime_type in ['text/plain', 'text/markdown']:
        logging.info(f"Reading text directly from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            extracted_text = file.read()
        if file_path.lower().endswith('.md'):
            already_in_markdown = True
            if 'email_bundle__sender__' in os.path.basename(file_path):
                metadata['is_email_bundle'] = True            
    elif detected_mime_type.startswith('application/pdf') and robust_needs_ocr(file_path):
        logging.info(f"PDF {file_path} requires OCR. Starting OCR process...")
        images = convert_pdf_to_images_ocr(file_path)
        with ThreadPoolExecutor() as executor:
            extracted_text_list = list(executor.map(ocr_image, images))
        extracted_text = await process_document_ocr(extracted_text_list)
        used_smart_ocr = True
    elif detected_mime_type.startswith('image/') or file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
        logging.info(f"Performing OCR on image {file_path}")
        extracted_text = perform_simple_ocr_on_image(file_path)
    elif file_extension in ['.docx', '.doc', '.rtf', '.odt', '.pptx', '.ppt', '.csv', '.xlsx', '.xls', '.html', '.htm']:
        logging.info(f"Extracting text from {file_extension} document {file_path} using textract")
        extracted_text = textract.process(file_path, encoding='utf-8').decode('utf-8')
    elif (file_extension == '.db' and 'sms' in file_path.lower()) or detected_mime_type == 'application/vnd.wap.mms-message':
        logging.info(f"Processing iPhone Messages database {file_path}")
        extracted_text = process_iphone_messages(file_path)
    elif (file_extension == '.db' and 'mmssms' in file_path.lower()) or detected_mime_type == 'application/vnd.android.mms': 
        logging.info(f"Processing Android Messages database {file_path}")
        extracted_text = process_android_messages(file_path)        
    elif detected_mime_type == 'message/rfc822':
        logging.info(f"Processing email file {file_path}")
        email_content = await parse_email_async(file_path)
        metadata = email_content['headers']
        extracted_text = f"From: {metadata['From']}\nTo: {metadata['To']}\nSubject: {metadata['Subject']}\nDate: {metadata['Date']}\n\n{email_content['body']}"
    else:
        logging.info(f"Attempting to extract text from {file_path} using textract") 
        try:
            extracted_text = textract.process(file_path, encoding='utf-8').decode('utf-8')
            logging.info(f"Successfully extracted text from {file_path} using textract")
        except Exception as e:
            logging.error(f"Error extracting text from {file_path} using textract: {str(e)}")
            extracted_text = f"Error extracting text from {file_path} using textract: {str(e)}"
    # Post-processing
    extracted_text = remove_pagination_breaks(extracted_text)
    sentences = sophisticated_sentence_splitter(extracted_text)
    processed_text = "\n".join(sentences)
    # Apply beautification and markdown formatting for non-OCR processed documents
    if not used_smart_ocr and not already_in_markdown:
        processed_text = await beautify_and_format_as_markdown(processed_text)
    # Save the processed text
    output_file_path = md_file_path if used_smart_ocr or already_in_markdown else txt_file_path
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    logging.info(f"Saved processed text to {output_file_path}")
    return processed_text, detected_mime_type, metadata, used_smart_ocr

async def convert_documents_to_plaintext(original_source_dir: str, converted_source_dir: str, files_to_convert: List[str]):
    logging.info("Starting conversion of source documents to plaintext")
    os.makedirs(converted_source_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(MAX_SOURCE_DOCUMENTS_TO_CONVERT_TO_PLAINTEXT_AT_ONE_TIME)
    async def process_file(file_path: str, pbar: tqdm):
        async with semaphore:
            if not os.path.isfile(file_path):
                pbar.set_postfix_str(f"Skipped {os.path.basename(file_path)} (not a file)")
                pbar.update(1)
                return
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            txt_file_path = os.path.join(converted_source_dir, f"{base_name}.txt")
            md_file_path = os.path.join(converted_source_dir, f"{base_name}.md")
            # Check if converted file already exists
            if os.path.exists(txt_file_path) or os.path.exists(md_file_path):
                logging.info(f"Skipping {os.path.basename(file_path)} - already converted ")
                pbar.set_postfix_str(f"Skipped {os.path.basename(file_path)} (already converted)")
                pbar.update(1)
                return
            try:
                pbar.set_postfix_str(f"Processing {os.path.basename(file_path)}")
                processed_text, mime_type, metadata, used_smart_ocr = await preprocess_document(file_path, converted_source_dir)
                if not processed_text.strip():
                    pbar.set_postfix_str(f"No text extracted from {os.path.basename(file_path)}")
                    pbar.update(1)
                    return
                converted_file_path = md_file_path if used_smart_ocr else txt_file_path
                with open(converted_file_path, 'w', encoding='utf-8') as f:
                    f.write(processed_text)
                file_size = os.path.getsize(converted_file_path)
                logging.info(f"Converted {os.path.basename(file_path)} ({mime_type}) - Size: {file_size / 1024:.2f} KB")
                pbar.set_postfix_str(f"Converted {os.path.basename(file_path)} ({mime_type})")
                pbar.update(1)
            except Exception as e:
                pbar.set_postfix_str(f"Error converting {os.path.basename(file_path)}: {str(e)}")
                pbar.update(1)
                logging.error(f"Error converting {os.path.basename(file_path)}: {str(e)}")
                logging.error(traceback.format_exc())
    logging.info(f"Found {len(files_to_convert)} files to process")
    with tqdm(total=len(files_to_convert), desc="Converting documents", unit="file") as pbar:
        tasks = [process_file(file_path, pbar) for file_path in files_to_convert]
        await asyncio.gather(*tasks)
    logging.info("Completed conversion of all documents to plaintext")
    use_delete_tiny_files = 0
    if use_delete_tiny_files:
        # Check for and remove tiny text files
        removed_files = 0
        for file_name in os.listdir(converted_source_dir):
            file_path = os.path.join(converted_source_dir, file_name)
            if file_name.endswith(('.txt', '.md')) and os.path.getsize(file_path) < 1024:  # Less than 1KB
                logging.warning(f"Removing tiny converted file: {file_path}")
                os.remove(file_path)
                removed_files += 1
        if removed_files > 0:
            logging.info(f"Removed {removed_files} tiny converted files (less than 1KB)")
        else:
            logging.info("No tiny converted files found")
    logging.info("Document conversion process completed")
    
async def check_corrupted_output_file(file_path: str, original_file_path: str, max_retries: int = 3) -> dict:
    """
    Check if the output file is likely corrupted or unusable due to failed OCR.
    Uses a more conservative approach to avoid false positives.
    
    :param file_path: Path to the processed output file
    :param original_file_path: Path to the original input file
    :param max_retries: Maximum number of retries for LLM analysis
    :return: Dictionary with corruption status and explanation
    """
    def parse_llm_response_for_corruption_check(response: str) -> dict:
        """Parse the LLM response into a dictionary."""
        analysis = {}
        for line in response.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                analysis[key.strip()] = value.strip()
        return analysis
    def validate_corruption_analysis_result(analysis: dict) -> bool:
        """Validate the parsed LLM response for corruption check."""
        required_keys = ['CORRUPTED', 'EXPLANATION', 'USABILITY_SCORE']
        if not all(key in analysis for key in required_keys):
            return False
        if analysis['CORRUPTED'].lower() not in ['yes', 'no']:
            return False
        try:
            usability_score = int(analysis['USABILITY_SCORE'])
            if not 0 <= usability_score <= 100:
                return False
        except ValueError:
            return False
        return True
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    prompt = f"""Carefully analyze the following text content from a processed document and determine if it's likely corrupted or unusable due to failed OCR. Be very conservative in your assessment - only flag as corrupted if there are clear and significant issues. Consider these factors:

1. Presence of a VERY high proportion of random characters or non-readable text
2. Complete lack of coherent sentences or recognizable words
3. Excessive and meaningless repetition of patterns or characters

Remember, the content might be fragmentary or contain specialized terms, so don't flag it as corrupted unless you're very certain. The bar for classifying as corrupted is VERY high, so if you're not sure, err on the side of caution.

Text content:
{content[:3000]}  # Limiting to first 3000 characters for brevity

Provide your analysis in the following format (without any additional text):
CORRUPTED: [Yes/No]
EXPLANATION: [Brief explanation of why the content is considered corrupted or not]
USABILITY_SCORE: [0-100, where 0 is completely unusable and 100 is perfect]
"""
    for attempt in range(max_retries):
        try:
            response = await generate_completion(prompt)
            analysis = parse_llm_response_for_corruption_check(response)
            if validate_corruption_analysis_result(analysis):
                is_corrupted = analysis['CORRUPTED'].lower() == 'yes'
                usability_score = int(analysis['USABILITY_SCORE'])
                # Additional safeguard: only consider it corrupted if usability is very low
                if is_corrupted and usability_score > 20:
                    is_corrupted = False
                    analysis['EXPLANATION'] += " However, the usability score suggests the content may still be valuable."
                return {
                    'input_file': original_file_path,
                    'output_file': file_path,
                    'is_corrupted': is_corrupted,
                    'explanation': analysis['EXPLANATION'],
                    'usability_score': usability_score
                }
            else:
                logging.warning(f"Invalid LLM response on attempt {attempt + 1}. Retrying...")
        except Exception as e:
            logging.error(f"Error on attempt {attempt + 1}: {str(e)}")
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    logging.error(f"Failed to get valid analysis after {max_retries} attempts")
    return {
        'input_file': original_file_path,
        'output_file': file_path,
        'is_corrupted': False,  # Default to not corrupted if analysis fails
        'explanation': 'Failed to analyze due to repeated errors, assuming not corrupted',
        'usability_score': 50  # Neutral score if analysis fails
    }
    
async def process_output_directory_to_check_for_corrupted_or_failed_files(output_dir: str, original_dir: str):
    logging.info(f"Starting to process directory: {output_dir}")
    semaphore = asyncio.Semaphore(25)
    processed_files = set()
    non_corrupt_files = set()
    async def process_file(filename: str, minimum_usability_score: int) -> Dict:
        async with semaphore:
            if filename.startswith('email_bundle__'):
                return None
            output_file_path = os.path.join(output_dir, filename)
            base_name = os.path.splitext(filename)[0].lower()
            if output_file_path in processed_files or output_file_path in non_corrupt_files:
                logging.info(f"Skipping already processed file: {filename}")
                return None
            original_file_path = next((os.path.join(original_dir, f) for f in os.listdir(original_dir) if os.path.splitext(f)[0].lower() == base_name), None)
            if not original_file_path:
                logging.info(f"Original file not found for converted file: {filename}. This may be normal if not all files have been converted yet, or if there is not a 1:1 mapping between original and converted files (as in the case of email archives).")
                return None
            try:
                result = await check_corrupted_output_file(output_file_path, original_file_path)
                logging.info(f"Processed {filename}: Corrupted: {result['is_corrupted']}, Usability: {result['usability_score']}")
                if not result['is_corrupted'] and result['usability_score'] >= minimum_usability_score:
                    non_corrupt_files.add(output_file_path)
                return result
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
                return {
                    'input_file': original_file_path,
                    'output_file': output_file_path,
                    'is_corrupted': True,
                    'explanation': f"Error during processing: {str(e)}",
                    'usability_score': 0
                }
    async def process_all_files() -> List[Dict]:
        tasks = [
            asyncio.create_task(process_file(filename, MINIMUM_OCR_USABILITY_SCORE))
            for filename in os.listdir(output_dir)
            if filename.lower().endswith(('.txt', '.md'))
        ]
        return [result for result in await asyncio.gather(*tasks) if result is not None]
    try:
        existing_corrupted_files = {}
        if os.path.exists(LIKELY_CORRUPTED_OUTPUT_FILES_JSON_PATH):
            with open(LIKELY_CORRUPTED_OUTPUT_FILES_JSON_PATH, 'r') as f:
                existing_corrupted_files = {file['output_file']: file for file in json.load(f)}
                processed_files.update(existing_corrupted_files.keys())
        if os.path.exists(LIST_OF_NON_CORRUPTED_FILES_PATH):
            with open(LIST_OF_NON_CORRUPTED_FILES_PATH, 'r') as f:
                non_corrupt_files.update(json.load(f))
        all_results = await process_all_files()
        updated_corrupted_files = {}
        for result in all_results:
            processed_files.add(result['output_file'])
            if result['is_corrupted'] or result['usability_score'] < MINIMUM_OCR_USABILITY_SCORE:
                updated_corrupted_files[result['output_file']] = result
            elif result['output_file'] in existing_corrupted_files:
                logging.info(f"File {result['output_file']} is no longer considered corrupted. Removing from list.")
        for file_path, file_data in existing_corrupted_files.items():
            if file_path not in updated_corrupted_files and os.path.exists(file_path):
                updated_corrupted_files[file_path] = file_data
        final_corrupted_files = list(updated_corrupted_files.values())
        with open(LIKELY_CORRUPTED_OUTPUT_FILES_JSON_PATH, 'w') as f:
            json.dump(final_corrupted_files, f, indent=2)
        with open(LIST_OF_NON_CORRUPTED_FILES_PATH, 'w') as f:
            json.dump(list(non_corrupt_files), f, indent=2)
        new_corrupted_count = sum(1 for file in final_corrupted_files if file['output_file'] not in existing_corrupted_files)
        removed_count = sum(1 for file in existing_corrupted_files if file not in updated_corrupted_files)
        logging.info(f"Identified {new_corrupted_count} new potentially corrupted or unusable files out of {len(all_results)} processed.")
        logging.info(f"Removed {removed_count} files that are no longer considered corrupted.")
        logging.info(f"Total corrupted files after update: {len(final_corrupted_files)}")
        logging.info(f"Results saved to {LIKELY_CORRUPTED_OUTPUT_FILES_JSON_PATH}")
        return final_corrupted_files
    except Exception as e:
        logging.error(f"An error occurred while processing the output directory: {str(e)}")
        return []
    
async def process_difficult_pdf_with_gpt4_vision(file_path: str) -> str:
    images = convert_pdf_to_images_ocr(file_path)
    all_text = []
    for i, img in enumerate(images):
        logging.info(f"Processing page {i+1} of {len(images)}")
        preprocessed_img = preprocess_image_ocr(img)
        buffered = io.BytesIO()
        preprocessed_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        prompt = f"""Please transcribe the text content from this image. This is page {i+1} of {len(images)} from a document.
        Maintain the original formatting as much as possible, including paragraphs, lists, and tables.
        Ignore any images or diagrams, focusing solely on the text content."""
        messages = [
            {"role": "system", "content": "You are a highly accurate OCR system."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
            ]}
        ]
        try:
            response = await api_request_with_retry(
                openai_client.chat.completions.create,
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=calculate_safe_max_tokens(1000, OPENAI_MAX_TOKENS),
                temperature=0.7,
            )
            page_text = response.choices[0].message.content
            all_text.append(page_text)
            logging.info(f"Successfully processed page {i+1}")
        except (RateLimitError, APIError) as e:
            logging.error(f"OpenAI API error on page {i+1}: {str(e)}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing page {i+1}: {str(e)}")
    return "\n\n--- New Page ---\n\n".join(all_text)

async def process_gpt4_vision_result(text: str) -> str:
    logging.info(f"Processing GPT-4 Vision result (length: {len(text):,} characters)")
    # Error correction template
    error_correction_template = """Review and correct any errors in the following text extracted from a document using GPT-4 Vision. Follow these guidelines:

1. Fix any obvious transcription errors or inconsistencies
2. Maintain original structure and formatting
3. Preserve all original content and meaning
4. Do not add any new information not present in the original text
5. Ensure proper paragraph breaks and formatting

IMPORTANT: Respond ONLY with the corrected text. Do not include any introduction, explanation, or metadata.

Text to process:
{text}

Corrected text:
"""

    # Markdown formatting template
    markdown_template = """Reformat the following text as markdown, improving readability while preserving the original structure. Follow these guidelines:
1. Convert headings to appropriate markdown heading levels (# for main titles, ## for subtitles, etc.)
2. Ensure each heading is on its own line with a blank line before and after
3. Maintain the original paragraph structure
4. Format lists properly (unordered or ordered) if they exist in the original text
5. Use emphasis (*italic*) and strong emphasis (**bold**) where appropriate
6. Preserve all original content and meaning
7. Do not add any extra punctuation or modify the existing punctuation
8. Do not add any introductory text, preamble, or markdown code block indicators

IMPORTANT: Start directly with the reformatted content.

Text to reformat:

{text}

Reformatted markdown:
"""
    # Final filtering template
    filtering_template = """Review the following markdown-formatted text and remove any invalid or unwanted elements without altering the actual content. Follow these guidelines:

1. Remove any markdown code block indicators (```) if present
2. Remove any preamble or introductory text
3. Ensure the text starts directly with the content (e.g., headings, paragraphs, or lists)
4. Do not remove any actual content, headings, or meaningful text
5. Preserve all markdown formatting (headings, lists, emphasis, etc.)
6. Remove any trailing whitespace or unnecessary blank lines at the end of the text

IMPORTANT: Only remove invalid elements as described above. Do not alter, summarize, or remove any of the actual content.

Text to filter:

{text}

Filtered text:
"""
    # Step 1: Error correction
    corrected_text = await process_text_with_llm(text, error_correction_template, max_chunk_size=1000)
    logging.info(f"Error correction completed. Output length: {len(corrected_text):,} characters")
    # Step 2: Markdown formatting
    markdown_formatted = await process_text_with_llm(corrected_text, markdown_template, max_chunk_size=1000)
    logging.info(f"Markdown formatting completed. Output length: {len(markdown_formatted):,} characters")
    # Step 3: Final filtering
    filtered_text = await process_text_with_llm(markdown_formatted, filtering_template, max_chunk_size=1000)
    logging.info(f"Final filtering completed. Output length: {len(filtered_text):,} characters")
    logging.info(f"GPT-4 Vision result processed. Final output length: {len(filtered_text):,} characters")
    return filtered_text

async def process_corrupted_files_with_gpt4_vision(corrupted_files_path: str):
    with open(corrupted_files_path, 'r') as f:
        corrupted_files = json.load(f)
    for file_info in corrupted_files:
        input_file = file_info['input_file']
        output_file = file_info['output_file']
        logging.info(f"Processing {input_file} with GPT-4 Vision API")
        try:
            extracted_text = await process_difficult_pdf_with_gpt4_vision(input_file)
            processed_text = await process_gpt4_vision_result(extracted_text)
            new_output_file = output_file.replace('.txt', '_gpt4vision.md').replace('.md', '_gpt4vision.md')
            with open(new_output_file, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            logging.info(f"Successfully processed {input_file}. Result saved to {new_output_file}")
        except Exception as e:
            logging.error(f"Error processing {input_file} with GPT-4 Vision API: {str(e)}")
            
def create_and_populate_case_sqlite_database(config_file_path, converted_source_dir, original_source_dir, discovery_output_dir):
    logging.info("Starting database creation and population process")
    db_dir = 'sqlite_database_files_of_converted_documents'
    os.makedirs(db_dir, exist_ok=True)
    config_base_name = os.path.splitext(os.path.basename(config_file_path))[0]
    db_file_path = os.path.join(db_dir, f"{config_base_name}.sqlite")
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    logging.info(f"Created SQLite database at: {db_file_path}")
    # Create tables
    tables = [
        ("case_metadata", '''CREATE TABLE IF NOT EXISTS case_metadata (
            id INTEGER PRIMARY KEY, config_file_path TEXT, freeform_input_text TEXT,
            json_config TEXT, case_name TEXT, creation_date TEXT)'''),
        ("documents", '''CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY, original_filename TEXT, converted_filename TEXT,
            file_hash TEXT UNIQUE, document_text TEXT, mime_type TEXT,
            file_size_bytes INTEGER, creation_date TEXT, last_modified_date TEXT,
            is_email BOOLEAN, email_from TEXT, email_to TEXT, email_subject TEXT,
            email_date TEXT, ocr_applied BOOLEAN, importance_score REAL,
            relevance_score REAL, processing_completion_datetime TEXT,
            processing_time_seconds REAL, llm_api_calls INTEGER,
            total_tokens_used INTEGER, high_priority_dossier_length INTEGER,
            low_priority_dossier_length INTEGER)'''),
        ("document_conversions", '''CREATE TABLE IF NOT EXISTS document_conversions (
            id INTEGER PRIMARY KEY, original_document_id INTEGER,
            converted_document_id INTEGER, conversion_type TEXT,
            FOREIGN KEY (original_document_id) REFERENCES documents (id),
            FOREIGN KEY (converted_document_id) REFERENCES documents (id))'''),
        ("discovery_goals", '''CREATE TABLE IF NOT EXISTS discovery_goals (
            id INTEGER PRIMARY KEY, description TEXT, importance INTEGER)'''),
        ("keywords", '''CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY, keyword TEXT UNIQUE, goal_id INTEGER,
            FOREIGN KEY (goal_id) REFERENCES discovery_goals (id))'''),
        ("entities_of_interest", '''CREATE TABLE IF NOT EXISTS entities_of_interest (
            id INTEGER PRIMARY KEY, entity_name TEXT UNIQUE)'''),
        ("document_entities", '''CREATE TABLE IF NOT EXISTS document_entities (
            document_id INTEGER, entity_id INTEGER,
            FOREIGN KEY (document_id) REFERENCES documents (id),
            FOREIGN KEY (entity_id) REFERENCES entities_of_interest (id),
            PRIMARY KEY (document_id, entity_id))'''),
        ("discovery_outputs", '''CREATE TABLE IF NOT EXISTS discovery_outputs (
            id INTEGER PRIMARY KEY, document_id INTEGER, output_type TEXT,
            output_content TEXT, FOREIGN KEY (document_id) REFERENCES documents (id))''')
    ]
    for table_name, create_table_sql in tables:
        cursor.execute(create_table_sql)
        logging.info(f"Created table: {table_name}")
    # Insert case metadata
    with open(config_file_path, 'r') as config_file:
        config_data = json.load(config_file)
    cursor.execute('''INSERT INTO case_metadata 
        (config_file_path, freeform_input_text, json_config, case_name, creation_date)
        VALUES (?, ?, ?, ?, ?)''', 
        (config_file_path, USER_FREEFORM_TEXT_GOAL_INPUT, json.dumps(config_data), 
            config_data['case_name'], datetime.now().isoformat()))
    logging.info("Inserted case metadata")
    # Insert discovery goals, keywords, and entities
    for goal in tqdm(config_data['discovery_goals'], desc="Inserting discovery goals and keywords"):
        cursor.execute('INSERT INTO discovery_goals (description, importance) VALUES (?, ?)',
                        (goal['description'], goal['importance']))
        goal_id = cursor.lastrowid
        for keyword in goal['keywords']:
            cursor.execute('INSERT OR IGNORE INTO keywords (keyword, goal_id) VALUES (?, ?)',
                            (keyword, goal_id))

    for entity in tqdm(config_data['entities_of_interest'], desc="Inserting entities of interest"):
        cursor.execute('INSERT OR IGNORE INTO entities_of_interest (entity_name) VALUES (?)',
                        (entity,))
    def insert_document(file_path, is_original):
        with open(file_path, 'rb') as file:
            file_content = file.read()
            result = magika.identify_bytes(file_content)
            detected_mime_type = result.output.mime_type
        file_hash = hashlib.sha256(file_content).hexdigest()
        is_text = detected_mime_type.startswith(('text/', 'application/json', 'application/xml'))
        if is_text:
            encodings_to_try = ['utf-8', 'latin-1', 'ascii', 'utf-16', 'utf-32']
            document_text = None
            for encoding in encodings_to_try:
                try:
                    document_text = file_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if document_text is None:
                is_text = False
        if not is_text:
            document_text = f"[Binary content, size: {len(file_content)} bytes, MIME type: {detected_mime_type}]"
        file_stats = os.stat(file_path)
        creation_date = datetime.fromtimestamp(file_stats.st_ctime).isoformat()
        last_modified_date = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        ocr_applied = file_path.endswith('.md')
        is_email = False
        email_from = email_to = email_subject = email_date = None
        if is_text and document_text.startswith("From:") and "To:" in document_text[:200]:
            is_email = True
            email_fields = ['From:', 'To:', 'Subject:', 'Date:']
            for field in email_fields:
                try:
                    value = document_text.split(field, 1)[1].split('\n', 1)[0].strip()
                    if field == 'From:':
                        email_from = value
                    elif field == 'To:':
                        email_to = value
                    elif field == 'Subject:':
                        email_subject = value
                    elif field == 'Date:':
                        email_date = value
                except IndexError:
                    logging.warning(f"Email field '{field}' not found in {file_path}")
        cursor.execute('''INSERT OR REPLACE INTO documents 
            (original_filename, converted_filename, file_hash, document_text,
            mime_type, file_size_bytes, creation_date, last_modified_date,
            is_email, email_from, email_to, email_subject, email_date,
            ocr_applied, importance_score, relevance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
            (os.path.basename(file_path), os.path.basename(file_path) if not is_original else None,
            file_hash, document_text, detected_mime_type, file_stats.st_size,
            creation_date, last_modified_date, is_email, email_from, email_to,
            email_subject, email_date, ocr_applied, 0, 0))
        return cursor.lastrowid, file_hash
    # Process original documents
    original_docs = {}
    original_files = [os.path.join(root, file) for root, _, files in os.walk(original_source_dir) for file in files]
    for file_path in tqdm(original_files, desc="Processing original documents"):
        doc_id, file_hash = insert_document(file_path, True)
        original_docs[file_hash] = doc_id

    # Process converted documents and link to original documents
    converted_files = [os.path.join(root, file) for root, _, files in os.walk(converted_source_dir) for file in files]
    for file_path in tqdm(converted_files, desc="Processing converted documents"):
        doc_id, file_hash = insert_document(file_path, False)
        original_doc_id = original_docs.get(file_hash)
        if original_doc_id:
            cursor.execute('''INSERT INTO document_conversions 
                (original_document_id, converted_document_id, conversion_type)
                VALUES (?, ?, ?)''', (original_doc_id, doc_id, 'direct'))
        else:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            if 'email_bundle' in base_name:
                cursor.execute('''INSERT INTO document_conversions 
                    (original_document_id, converted_document_id, conversion_type)
                    VALUES (?, ?, ?)''', (None, doc_id, 'email_bundle'))
            elif base_name.startswith('pst_'):
                cursor.execute('''INSERT INTO document_conversions 
                    (original_document_id, converted_document_id, conversion_type)
                    VALUES (?, ?, ?)''', (None, doc_id, 'pst_extraction'))

    # Process discovery outputs
    processed_files_dir = os.path.join(discovery_output_dir, 'processed_files')
    os.makedirs(processed_files_dir, exist_ok=True)
    processed_files = [f for f in os.listdir(processed_files_dir) if f.endswith('.json')]
    for filename in tqdm(processed_files, desc="Processing discovery outputs"):
        with open(os.path.join(processed_files_dir, filename), 'r') as f:
            processing_info = json.load(f)
        converted_filename = os.path.splitext(filename)[0]
        cursor.execute('SELECT id FROM documents WHERE converted_filename = ?', (converted_filename,))
        document_id = cursor.fetchone()
        if document_id:
            document_id = document_id[0]
            cursor.execute('''UPDATE documents SET
                processing_completion_datetime = ?,
                processing_time_seconds = ?,
                llm_api_calls = ?,
                total_tokens_used = ?,
                high_priority_dossier_length = ?,
                low_priority_dossier_length = ?
                WHERE id = ?''', 
                (processing_info['completion_datetime'],
                processing_info['processing_time_seconds'],
                processing_info['llm_api_calls'],
                processing_info['total_tokens_used'],
                processing_info['high_priority_dossier_length'],
                processing_info['low_priority_dossier_length'],
                document_id))

    # Process other discovery outputs
    results_dir = os.path.join(discovery_output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    result_files = [f for f in os.listdir(results_dir) if f.endswith('_results.json')]
    for filename in tqdm(result_files, desc="Processing discovery results"):
        with open(os.path.join(results_dir, filename), 'r') as f:
            result_data = json.load(f)
        converted_filename = result_data.get('file_path')
        if converted_filename:
            cursor.execute('SELECT id FROM documents WHERE converted_filename = ?', (os.path.basename(converted_filename),))
            document_id = cursor.fetchone()
            if document_id:
                document_id = document_id[0]
                if 'dossier_section' in result_data:
                    cursor.execute('''INSERT INTO discovery_outputs 
                        (document_id, output_type, output_content)
                        VALUES (?, ?, ?)''', 
                        (document_id, 'high_priority_dossier', result_data['dossier_section']))
                if 'low_importance_section' in result_data:
                    cursor.execute('''INSERT INTO discovery_outputs 
                        (document_id, output_type, output_content)
                        VALUES (?, ?, ?)''', 
                        (document_id, 'low_priority_dossier', result_data['low_importance_section']))
    conn.commit()
    conn.close()
    logging.info(f"Enhanced SQLite database created and populated at: {db_file_path}")
                
async def determine_files_to_process(
    original_source_dir: str,
    converted_source_dir: str,
    discovery_output_dir: str
) -> Tuple[List[str], List[str]]:
    files_to_convert: Set[str] = set()
    files_to_discover: Set[str] = set()
    processed_files_dir = os.path.join(discovery_output_dir, 'processed_files')
    os.makedirs(processed_files_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(1000)  # Adjust this value based on your system's capabilities
    async def check_file(file_path: str, is_original: bool, pbar: tqdm):
        async with semaphore:
            relative_path = os.path.relpath(file_path, original_source_dir if is_original else converted_source_dir)
            base_name = os.path.splitext(relative_path)[0]
            if is_original:
                if file_path.lower().endswith('.pst'):
                    pst_converted_dir = os.path.join(converted_source_dir, base_name)
                    if not os.path.exists(pst_converted_dir) or not os.listdir(pst_converted_dir):
                        files_to_convert.add(file_path)
                        pbar.set_description(f"Needs conversion: {relative_path}")
                        logging.info(f"PST file needs conversion: {relative_path}")
                    else:
                        pbar.set_description(f"Already converted: {relative_path}")
                        logging.info(f"PST file already converted: {relative_path}")
                else:
                    converted_txt_path = os.path.join(converted_source_dir, f"{base_name}.txt")
                    converted_md_path = os.path.join(converted_source_dir, f"{base_name}.md")
                    if not os.path.exists(converted_txt_path) and not os.path.exists(converted_md_path):
                        files_to_convert.add(file_path)
                        pbar.set_description(f"Needs conversion: {relative_path}")
                        logging.info(f"File needs conversion: {relative_path}")
                    else:
                        pbar.set_description(f"Already converted: {relative_path}")
                        logging.info(f"File already converted: {relative_path}")
            else:
                if file_path.endswith(('.txt', '.md')):
                    processed_file_path = os.path.join(processed_files_dir, f"{base_name}.json")
                    if not os.path.exists(processed_file_path):
                        files_to_discover.add(file_path)
                        pbar.set_description(f"Needs discovery: {relative_path}")
                        logging.info(f"File needs discovery processing: {relative_path}")
                    else:
                        pbar.set_description(f"Already processed: {relative_path}")
                        logging.info(f"Skipping already processed file: {relative_path}")
            pbar.update(1)
    all_files = []
    for root, _, files in os.walk(original_source_dir):
        all_files.extend((os.path.join(root, filename), True) for filename in files)
    for root, _, files in os.walk(converted_source_dir):
        all_files.extend((os.path.join(root, filename), False) for filename in files)
    logging.info(f"Total files to check: {len(all_files)}")
    with tqdm(total=len(all_files), desc="Checking files", unit="file") as pbar:
        tasks = [check_file(file_path, is_original, pbar) for file_path, is_original in all_files]
        await asyncio.gather(*tasks)
    logging.info(f"Files to convert: {len(files_to_convert)}")
    logging.info(f"Files to discover: {len(files_to_discover)}")
    return list(files_to_convert), list(files_to_discover)

async def process_document_async(file_path: str, converted_source_dir: str, discovery_params: Dict[str, Any], discovery_output_dir: str, semaphore: asyncio.Semaphore, model_name: str, max_chunk_tokens: int):
    logging.info(f"Starting processing of file: {file_path}")
    start_time = time.time()
    total_api_calls = 0
    total_tokens_used = 0
    with open(file_path, 'rb') as file:
        file_content = file.read()
        result = magika.identify_bytes(file_content)
        detected_mime_type = result.output.mime_type
    document_id = hashlib.sha256(file_content).hexdigest()
    json_output_path = os.path.join(discovery_output_dir, f"{document_id}_results.json")
    result = {
        'document_id': document_id,
        'file_path': file_path,
        'processed_chunks': [],
        'low_importance_chunks': [],
        'metadata': {
            'file_path': file_path,
            'mime_type': detected_mime_type,
            'file_size_bytes': os.path.getsize(file_path)
        }
    }
    logging.info(f"File {file_path}: Detected MIME type: {detected_mime_type}, Size: {result['metadata']['file_size_bytes']} bytes")
    # Save initial result
    with open(json_output_path, 'w') as f:
        json.dump(result, f, indent=2)
    logging.info(f"File {file_path}: Initial result saved to {json_output_path}")
    try:
        processed_text, _, metadata, _ = await preprocess_document(file_path, converted_source_dir)
        sentences = sophisticated_sentence_splitter(processed_text)
        result['metadata']['total_sentences'] = len(sentences)
        result['metadata']['thousands_of_input_words'] = round(sum(len(s.split()) for s in sentences) / 1000, 2)
        if metadata:
            result['metadata'].update(metadata)
        logging.info(f"File {file_path}: Preprocessed. Total sentences: {result['metadata']['total_sentences']}, Estimated words: {result['metadata']['thousands_of_input_words']}K")
        # Update result file with metadata
        with open(json_output_path, 'w') as f:
            json.dump(result, f, indent=2)
        chunk_size = 10
        chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
        logging.info(f"File {file_path}: Split into {len(chunks)} chunks for processing")
        async def process_chunk(chunk, i):
            async with semaphore:
                chunk_text = " ".join(chunk)
                logging.info(f"File {file_path}: Processing chunk {i+1}/{len(chunks)}")
                chunk_result, api_calls, tokens_used = await process_chunk_multi_stage(chunk_text, i, len(chunks), discovery_params)
                if chunk_result:
                    importance_score = float(chunk_result['importance_score']['final_score'])
                    if importance_score >= discovery_params['minimum_importance_score']:
                        result['processed_chunks'].append(chunk_result)
                        logging.info(f"File {file_path}: Chunk {i+1}/{len(chunks)} added to high-importance chunks. Score: {importance_score:.2f}")
                    else:
                        result['low_importance_chunks'].append(chunk_result)
                        logging.info(f"File {file_path}: Chunk {i+1}/{len(chunks)} added to low-importance chunks. Score: {importance_score:.2f}")
                    logging.info(f"File {file_path}: Processed chunk {i+1}/{len(chunks)}. Importance score: {importance_score:.2f}. Total high-importance chunks: {len(result['processed_chunks'])}, Total low-importance chunks: {len(result['low_importance_chunks'])}")
                    # Update result file after each chunk
                    with open(json_output_path, 'w') as f:
                        json.dump(result, f, indent=2)
                return chunk_result, api_calls, tokens_used
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            chunk_results = await asyncio.gather(*[loop.run_in_executor(executor, lambda: asyncio.run(process_chunk(chunk, i))) for i, chunk in enumerate(chunks)])
        for chunk_result, api_calls, tokens_used in chunk_results:
            if chunk_result:
                total_api_calls += api_calls
                total_tokens_used += tokens_used
        logging.info(f"File {file_path}: All chunks processed. Total API calls: {total_api_calls}, Total tokens used: {total_tokens_used}")
        if not result['processed_chunks'] and not result['low_importance_chunks']:
            logging.info(f"File {file_path}: Did not yield any relevant information.")
            return None
        logging.info(f"File {file_path}: Compiling dossier sections")
        dossier_section = compile_dossier_section(result['processed_chunks'], document_id, file_path, discovery_params)
        low_importance_section = compile_dossier_section(result['low_importance_chunks'], document_id, file_path, discovery_params)
        result['dossier_section'] = dossier_section['dossier_section'] if dossier_section else None
        result['importance_score'] = dossier_section['importance_score'] if dossier_section else 0
        result['low_importance_section'] = low_importance_section['dossier_section'] if low_importance_section else None
        logging.info(f"File {file_path}: Dossier sections compiled. Overall importance score: {result['importance_score']:.2f}")
        end_time = time.time()
        result['processing_time_seconds'] = end_time - start_time
        result['total_api_calls'] = total_api_calls
        result['total_tokens_used'] = total_tokens_used
        # Save final result
        with open(json_output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logging.info(f"File {file_path}: Final result saved to {json_output_path}")
        # Generate document-level dossier
        dossier_file_name = f"document_level_dossier__{os.path.splitext(os.path.basename(file_path))[0]}.md"
        dossier_file_path = os.path.join(discovery_output_dir, dossier_file_name)
        with open(dossier_file_path, 'w', encoding='utf-8') as f:
            if result['dossier_section']:
                f.write(result['dossier_section'])
            if result['low_importance_section']:
                f.write("\n\n## Low Importance Section\n\n")
                f.write(result['low_importance_section'])
        logging.info(f"File {file_path}: Document-level dossier generated and saved to {dossier_file_path}")
        logging.info(f"File {file_path}: Processing completed in {result['processing_time_seconds']:.2f} seconds. "
                        f"API calls: {total_api_calls}, Tokens used: {total_tokens_used}, "
                        f"Importance score: {result['importance_score']:.2f}")
        return result
    except Exception as e:
        logging.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        logging.debug(f"Full traceback for {os.path.basename(file_path)}:", exc_info=True)
        return None

def compile_results(file_path: str, discovery_params: Dict[str, Any]) -> Dict[str, Any]:
    json_output_path = file_path.replace('.txt', '_discovery_results.json').replace('.md', '_discovery_results.json')
    logging.info(f"Compiling results for file: {file_path}")
    if not os.path.exists(json_output_path):
        logging.error(f"Results file not found for {file_path}")
        return None
    with open(json_output_path, 'r') as f:
        document_results = json.load(f)
    compiled_result = {
        'file_path': file_path,
        'document_id': hashlib.sha256(file_path.encode()).hexdigest(),
        'dossier_section': '',
        'low_importance_section': '',
        'importance_score': 0,
        'processed_chunks': document_results['processed_chunks'],
        'metadata': {
            'file_path': file_path,
            'total_chunks': len(document_results['processed_chunks']),
            'total_api_calls': document_results['total_api_calls'],
            'total_tokens_used': document_results['total_tokens_used'],
            'processing_time_seconds': document_results.get('processing_time_seconds', 0)
        }
    }
    logging.info(f"File {file_path}: Loaded {compiled_result['metadata']['total_chunks']} processed chunks")
    # Combine dossier sections and calculate overall importance score
    high_importance_chunks = []
    low_importance_chunks = []
    for chunk_result in document_results['processed_chunks']:
        if chunk_result['importance_score']['final_score'] >= discovery_params['minimum_importance_score']:
            high_importance_chunks.append(chunk_result)
        else:
            low_importance_chunks.append(chunk_result)
    logging.info(f"File {file_path}: Identified {len(high_importance_chunks)} high-importance chunks and {len(low_importance_chunks)} low-importance chunks")
    if document_results['processed_chunks']:
        compiled_result['importance_score'] = sum(chunk['importance_score']['final_score'] for chunk in document_results['processed_chunks']) / len(document_results['processed_chunks'])
        logging.info(f"File {file_path}: Calculated overall importance score: {compiled_result['importance_score']:.2f}")
    else:
        logging.warning(f"File {file_path}: No processed chunks found, importance score set to 0")
    if high_importance_chunks:
        compiled_result['dossier_section'] = compile_dossier_section(high_importance_chunks, compiled_result['document_id'], file_path, discovery_params)['dossier_section']
        logging.info(f"File {file_path}: Compiled high-importance dossier section")
    else:
        logging.info(f"File {file_path}: No high-importance chunks to compile")
    if low_importance_chunks:
        compiled_result['low_importance_section'] = compile_dossier_section(low_importance_chunks, compiled_result['document_id'], file_path, discovery_params)['dossier_section']
        logging.info(f"File {file_path}: Compiled low-importance dossier section")
    else:
        logging.info(f"File {file_path}: No low-importance chunks to compile")
    # Save document-level dossier
    dossier_file_name = f"document_level_dossier__{os.path.splitext(os.path.basename(file_path))[0]}.md"
    dossier_file_path = os.path.join(os.path.dirname(json_output_path), dossier_file_name)
    with open(dossier_file_path, 'w', encoding='utf-8') as f:
        f.write(compiled_result['dossier_section'])
        if compiled_result['low_importance_section']:
            f.write("\n\n## Low Importance Section\n\n")
            f.write(compiled_result['low_importance_section'])
    logging.info(f"File {file_path}: Saved document-level dossier to {dossier_file_path}")
    logging.info(f"File {file_path}: Compilation complete. Total chunks: {compiled_result['metadata']['total_chunks']}, "
                    f"API calls: {compiled_result['metadata']['total_api_calls']}, "
                    f"Tokens used: {compiled_result['metadata']['total_tokens_used']}, "
                    f"Processing time: {compiled_result['metadata']['processing_time_seconds']:.2f} seconds")
    return compiled_result

def process_document_wrapper_mp(file_path, converted_source_dir, discovery_params, discovery_output_dir, semaphore, model_name, max_chunk_tokens):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(process_document_async(file_path, converted_source_dir, discovery_params, discovery_output_dir, semaphore, model_name, max_chunk_tokens))
        return result
    finally:
        loop.close()


#############################################################################################################################    
    
# Static configuration
MAX_SOURCE_DOCUMENTS_TO_CONVERT_TO_PLAINTEXT_AT_ONE_TIME = 15
USE_GPT4_VISION_MODEL_FOR_FAILED_OR_CORRUPTED_FILES = 0
USE_OVERRIDE_DISCOVERY_CONFIG_JSON_FILE = 1  # Set to 1 to use override file
OVERRIDE_CONFIG_FILE_PATH = "discovery_configuration_json_files/shareholders_vs_enron_corporation.json"
MINIMUM_OCR_USABILITY_SCORE = 20 # Below this threshold, the document will be considered corrupted
LIKELY_CORRUPTED_OUTPUT_FILES_JSON_PATH = 'likely_corrupted_output_files.json'
LIST_OF_NON_CORRUPTED_FILES_PATH = "converted_output_files_already_checked_for_corruption_that_were_not_corrupted.json"

# This is just a simple example of a free-form text goal input; this will be overwritten with the Enron-specific input if use_enron_example is set to 1
USER_FREEFORM_TEXT_GOAL_INPUT = """
We're working on a case involving patent infringement by TechCorp against our client, InnovativeTech. 
We need to find any communications or documents that discuss TechCorp's knowledge of our client's patent 
before they released their competing product. We're also interested in any internal documents from TechCorp 
that discuss the development of their product, especially if they mention our client or their patent. 
The key people we're interested in are John Smith (TechCorp's CTO) and Sarah Johnson (TechCorp's lead engineer on the project).
We want to focus on documents from the last 5 years, and only include highly relevant documents in our final report.
"""

use_enron_example = 1

if use_enron_example:
    use_more_basic_freeform_text_goal_input = 0
    
    logging.info("Using Enron example, so replacing free-form text goal input with Enron-specific input.")
    
    if use_more_basic_freeform_text_goal_input:
        USER_FREEFORM_TEXT_GOAL_INPUT = """
        We're representing shareholders in a lawsuit against Enron Corporation and its key executives. Our clients suspect fraudulent accounting practices and misrepresentation of the company's financial health. We need to conduct a thorough investigation of Enron's internal communications and financial documents from the past 5 years.

        Key discovery goals:
        1. Uncover evidence of deliberate financial misreporting or fraudulent accounting practices.
        2. Identify communications discussing the creation or use of off-book entities to hide debt or inflate profits.
        3. Find instances where executives acknowledged financial problems while publicly claiming the company was healthy.
        4. Discover any discussions about manipulating energy markets or prices.
        5. Locate evidence of insider trading by executives or their associates.
        6. Identify any attempts to pressure or influence auditors (Arthur Andersen) to approve questionable financial statements.
        7. Find communications discussing the risks or potential consequences of their accounting practices.
        8. Uncover any evidence of document shredding or attempts to hide information.

        We're particularly interested in communications involving key executives such as Kenneth Lay (CEO), Jeffrey Skilling (COO/CEO), Andrew Fastow (CFO), and Richard Causey (Chief Accounting Officer). Other persons of interest include executives at Arthur Andersen and any mention of financial analysts who were skeptical of Enron's reported profits.

        Focus on documents that discuss Special Purpose Entities (SPEs), mark-to-market accounting, the Raptors, LJM partnerships, and any mentions of "aggressive" accounting practices. We're also interested in any discussions about the company's stock price, especially in relation to executive compensation or stock options.

        Prioritize highly relevant documents that clearly demonstrate knowledge of wrongdoing or attempts to deceive shareholders, auditors, or regulators. We're looking for smoking guns - clear admissions of guilt, discussions of illegal activities, or explicit concerns about the legality or ethics of their practices.

        The final report should provide a clear narrative of how Enron's executives deliberately misled shareholders and manipulated financial statements, supported by the most damning evidence found in the documents.
        """
    else:
        USER_FREEFORM_TEXT_GOAL_INPUT = """
        Case Name: Shareholders vs. Enron Corporation

        Summary:
        We are representing shareholders in a lawsuit against Enron Corporation and its key executives. Our clients suspect fraudulent accounting practices and misrepresentation of the company's financial health. The investigation will focus on Enron's internal communications and financial documents from the past 5 years.

        Discovery Goals:
        1. Uncover evidence of deliberate financial misreporting or fraudulent accounting practices.
        - Keywords: financial misreporting, fraud, accounting practices
        - Importance: 10

        2. Identify communications discussing the creation or use of off-book entities to hide debt or inflate profits.
        - Keywords: off-book entities, hide debt, inflate profits
        - Importance: 9

        3. Find instances where executives acknowledged financial problems while publicly claiming the company was healthy.
        - Keywords: executives acknowledgment, financial problems, public claims
        - Importance: 8

        4. Discover any discussions about manipulating energy markets or prices.
        - Keywords: manipulating markets, energy prices, market manipulation
        - Importance: 7

        5. Locate evidence of insider trading by executives or their associates.
        - Keywords: insider trading, executives, associates
        - Importance: 8

        6. Identify any attempts to pressure or influence auditors (Arthur Andersen) to approve questionable financial statements.
        - Keywords: pressure auditors, Arthur Andersen, questionable statements
        - Importance: 9

        7. Find communications discussing the risks or potential consequences of their accounting practices.
        - Keywords: risks, accounting practices, consequences
        - Importance: 6

        8. Uncover any evidence of document shredding or attempts to hide information.
        - Keywords: document shredding, hide information, destruction of evidence
        - Importance: 8

        Entities of Interest:
        - Kenneth Lay (CEO)
        - Jeffrey Skilling (COO/CEO)
        - Andrew Fastow (CFO)
        - Richard Causey (Chief Accounting Officer)
        - Arthur Andersen (auditors)
        - Financial analysts skeptical of Enron’s profits

        Minimum Importance Score:
        - 70.0
        """

async def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define directories
    project_root = os.path.dirname(os.path.abspath(__file__))
    original_source_dir = os.path.join(project_root, 'folder_of_source_documents__original_format')
    converted_source_dir = os.path.join(project_root, 'folder_of_source_documents__converted_to_plaintext')
    config_dir = os.path.join(project_root, 'discovery_configuration_json_files')

    # Ensure directories exist
    os.makedirs(original_source_dir, exist_ok=True)
    os.makedirs(converted_source_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # Step 1: Generate or load configuration
    if USE_OVERRIDE_DISCOVERY_CONFIG_JSON_FILE:
        config_file_path = OVERRIDE_CONFIG_FILE_PATH
        logging.info(f"Loading existing configuration from: {config_file_path}")
        with open(config_file_path, 'r') as f:
            discovery_params = json.load(f)
    else:
        logging.info("Generating configuration from user input")
        config_file_path = await generate_discovery_config(USER_FREEFORM_TEXT_GOAL_INPUT)
        with open(config_file_path, 'r') as f:
            discovery_params = json.load(f)

    # Set up session-specific logging
    log_file_name = os.path.splitext(os.path.basename(config_file_path))[0] + '.log'
    log_file_path = os.path.join(project_root, log_file_name)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)

    # Download Enron sample PDF exhibit files
    if use_enron_example:
        logging.info("Now downloading miscellaneous Enron related exhibits as PDFs...")
        await enron_collector_main()
        logging.info("Enron sample data collection completed.")
        
    # Create a discovery_output_dir folder for intermediate results
    discovery_output_dir = os.path.join(project_root, f"discovery_output__{os.path.splitext(os.path.basename(config_file_path))[0]}")
    os.makedirs(discovery_output_dir, exist_ok=True)
    
    # Determine which files need to be processed
    files_to_convert, files_to_discover = await determine_files_to_process(original_source_dir, converted_source_dir, discovery_output_dir)

    use_skip_conversion = 0
    if use_skip_conversion:
        logging.info("Skipping conversion of source documents to plaintext...")
    else:
        # Step 2a: Convert source documents to plaintext
        # Process MS Outlook PST files
        logging.info("Now processing MS Outlook PST files...")
        await process_pst_files(original_source_dir, converted_source_dir)
        logging.info("Now converting source documents to plaintext (and performing OCR if needed)...")
        await convert_documents_to_plaintext(original_source_dir, converted_source_dir, files_to_convert)
    
    # Step 2b: Check for corrupted/failed output files
    logging.info("Now checking for corrupted or failed output files...")
    await process_output_directory_to_check_for_corrupted_or_failed_files(converted_source_dir, original_source_dir)
    
    # Load and process the JSON file containing corrupted or failed files
    with open(LIKELY_CORRUPTED_OUTPUT_FILES_JSON_PATH, 'r') as f:
        corrupted_files_data = json.load(f)
    list_of_corrupted_or_failed_files = [item['output_file'] for item in corrupted_files_data]
    total_size = sum(os.path.getsize(file_path) for file_path in list_of_corrupted_or_failed_files if os.path.exists(file_path))
    logging.info(f"Found {len(list_of_corrupted_or_failed_files)} corrupted or failed output files, with a total data size of {total_size / 1024 / 1024:.4f} MB")
        
    use_delete_corrupted_or_failed_files = 0
    if use_delete_corrupted_or_failed_files:
        # Delete corrupted or failed output files
        logging.info(f"Deleting {len(list_of_corrupted_or_failed_files)} corrupted or failed output files...")
        for file_path in list_of_corrupted_or_failed_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024 / 1024  # Size in MB
                logging.info(f"Deleting corrupted or failed output file: {file_path} (size: {file_size:.4f} MB)")
                os.remove(file_path)
            else:
                logging.warning(f"File not found, cannot delete: {file_path}")
    else:
        logging.info("Skipping deletion of corrupted or failed output files...")
    
    if USE_GPT4_VISION_MODEL_FOR_FAILED_OR_CORRUPTED_FILES:
        # Step 2b.1: Process corrupted/failed output files with GPT-4 Vision instead of pytesseract
        logging.info("Now processing corrupted or failed output files that did not work well with pytesseract with GPT-4 Vision... (more expensive but more capable with difficult files like handwritten documents)")
        await process_corrupted_files_with_gpt4_vision(LIKELY_CORRUPTED_OUTPUT_FILES_JSON_PATH)
    else:
        logging.info("Skipping processing of corrupted or failed output files with GPT-4 Vision...")

    # Step 2c: Process Enron email corpus, turning the archive into individual markdown files per sender if desired
    if use_enron_example:
        await process_enron_email_corpus(
            project_root,
            original_source_dir,
            converted_source_dir
        )          
            
    use_create_sqlite_database = 0
    if use_create_sqlite_database:
        # Step 2d: Create and populate a SQLite database for the discovery case
        logging.info("Creating and populating SQLite database containing converted documents and their metadata...")
        create_and_populate_case_sqlite_database(config_file_path, converted_source_dir, original_source_dir, discovery_output_dir)
    else:
        logging.info("Skipping creation of SQLite database...")
    
    # Create a semaphore to limit concurrent API calls
    MAX_CONCURRENT_CONVERTED_DOCUMENTS_TO_PROCESS_FOR_DISCOVERY = 1  # Adjust this value based on API rate limits
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CONVERTED_DOCUMENTS_TO_PROCESS_FOR_DISCOVERY)

    # Step 3: Process converted documents
    logging.info("\n________________________________________________________________________________\n\nNow performing automated legal discovery on converted documents!!")

    model_name = OPENAI_COMPLETION_MODEL if API_PROVIDER == "OPENAI" else CLAUDE_MODEL_STRING
    max_chunk_tokens = OPENAI_MAX_TOKENS - TOKEN_BUFFER - max(
        estimate_tokens(doc_id_prompt, model_name),
        estimate_tokens(relevance_check_prompt, model_name),
        estimate_tokens(extract_gen_prompt, model_name),
        estimate_tokens(tag_gen_prompt, model_name),
        estimate_tokens(explanation_gen_prompt, model_name),
        estimate_tokens(importance_score_prompt, model_name)
    )

    # Use multiprocessing to distribute document processing with a progress bar
    total_files = len(files_to_discover)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        with tqdm(total=total_files, desc="Processing documents", unit="file") as pbar:
            results = []
            for result in pool.imap_unordered(
                partial(process_document_wrapper_mp, 
                        converted_source_dir=converted_source_dir,
                        discovery_params=discovery_params, 
                        discovery_output_dir=discovery_output_dir,
                        semaphore=semaphore,
                        model_name=model_name,
                        max_chunk_tokens=max_chunk_tokens),
                files_to_discover
            ):
                if result is not None:
                    results.append(result)
                pbar.update()

    # Compile complete case level dossiers from saved JSON files
    dossier_sections = []
    low_importance_sections = []
    results_dir = os.path.join(discovery_output_dir, 'results')
    for file in os.listdir(results_dir):
        if file.endswith('_results.json'):
            try:
                with open(os.path.join(results_dir, file), 'r') as f:
                    data = json.load(f)
                    if data.get('dossier_section'):
                        dossier_sections.append(data['dossier_section'])
                    if data.get('low_importance_section'):
                        low_importance_sections.append(data['low_importance_section'])
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON file: {file}")
            except Exception as e:
                logging.error(f"Error processing file {file}: {str(e)}")

    # Step 4: Compile final dossier
    logging.info("Compiling final dossier")
    final_dossier = compile_final_dossier(dossier_sections)

    # Step 5: Compile low-importance dossier
    logging.info("Compiling low-importance dossier")
    low_importance_dossier = compile_final_dossier(low_importance_sections)

    # Step 6: Write final dossiers to files
    dossier_file_name = os.path.splitext(os.path.basename(config_file_path))[0] + '_dossier.md'
    dossier_file_path = os.path.join(project_root, dossier_file_name)
    with open(dossier_file_path, 'w', encoding='utf-8') as f:
        f.write(final_dossier)

    low_importance_file_name = os.path.splitext(os.path.basename(config_file_path))[0] + '_low_importance_dossier.md'
    low_importance_file_path = os.path.join(project_root, low_importance_file_name)
    with open(low_importance_file_path, 'w', encoding='utf-8') as f:
        f.write(low_importance_dossier)

    logging.info(f"Legal discovery process completed. Dossier saved to {dossier_file_path}")
    logging.info(f"Low-importance dossier saved to {low_importance_file_path}")

if __name__ == '__main__':
    asyncio.run(main())