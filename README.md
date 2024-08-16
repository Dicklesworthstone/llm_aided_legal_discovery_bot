# LLM-Aided Legal Discovery Automation

## Introduction

LLM-Aided Legal Discovery Automation is a powerful tool designed to streamline the legal discovery process by leveraging advanced AI models. It addresses the challenges legal professionals face when dealing with large volumes of documents in cases such as corporate litigation, intellectual property disputes, or regulatory investigations.

Traditional legal discovery processes often require manual review of thousands or millions of documents, which is time-consuming, labor-intensive, and prone to human error. This tool transforms the process through several key innovations:

1. **Flexible Goal Setting**: Users can define discovery goals either through a structured JSON template or by providing a free-form text description. The system can automatically generate a structured configuration file from the latter using AI interpretation.

2. **Diverse Document Handling**: The system processes a wide range of document formats, including PDFs (both scanned and digital), Word documents, plain text files, HTML, Outlook PST archives, and even mobile device message databases (iPhone and Android).

3. **Advanced OCR Capabilities**: For scanned documents or images, the tool employs Tesseract OCR, with an optional fallback to GPT-4 Vision API for particularly challenging documents.

4. **Intelligent Document Analysis**: Each document undergoes multi-stage AI-powered analysis to assess relevance to specified discovery goals, extract key information, and generate summaries.

5. **Comprehensive Dossier Generation**: The tool produces detailed dossiers of relevant information, including document summaries, relevance explanations, and importance scores.

6. **Efficient Data Management**: A SQLite database is used to store and manage processed documents, enabling quick retrieval and full-text search capabilities.

7. **Incremental Processing**: The system tracks processed files, allowing for efficient updates when new documents are added or existing ones are modified.

8. **Performance Optimization**: Parallel processing and asynchronous operations are employed to handle large document sets efficiently.

9. **Flexible AI Integration**: The tool supports multiple AI providers (OpenAI, Anthropic's Claude) and local LLMs, allowing users to choose based on their needs for privacy, cost, or specific capabilities.

10. **Detailed Logging and Error Handling**: Comprehensive logs are maintained for auditing and debugging purposes, ensuring transparency and reproducibility of the discovery process.

By automating the initial stages of document review and analysis, this tool allows legal professionals to focus on high-level strategy and decision-making. It's designed to handle cases of varying sizes and complexities, from small internal investigations to large-scale multi-national litigations.

The system's flexibility and scalability make it particularly valuable for law firms, corporate legal departments, and legal technology companies dealing with document-intensive cases. By leveraging AI and efficient data processing techniques, it aims to significantly reduce the time and cost associated with legal discovery while potentially uncovering insights that might be missed in manual review processes.

## Key Features

- Automated document processing and conversion for various file formats (PDF, DOC, DOCX, TXT, HTML, etc.)
- OCR capability for scanned documents using Tesseract
- AI-powered document analysis using OpenAI or Anthropic's Claude API
- Option to use local LLM for enhanced privacy
- Customizable discovery goals and parameters through JSON configuration
- Relevance assessment and importance scoring for documents
- Extraction of key information and generation of document summaries
- Creation of comprehensive dossiers for relevant documents
- Support for processing Outlook PST files and extracting emails
- Handling of iPhone and Android message databases
- Incremental processing with tracking of already processed files
- Parallel processing for improved performance
- SQLite database integration for efficient document management
- Full-text search capabilities in the SQLite database
- Enron email corpus processing for testing and demonstration
- GPT-4 Vision API integration for difficult-to-OCR documents (optional)
- Detailed logging and error handling
- Rate limiting and retry logic for API calls

## How It Works: High-Level Overview

1. **Configuration**: The system generates or loads a JSON configuration file defining discovery goals, keywords, and entities of interest. It can create this from free-form user input using AI interpretation.

2. **Document Preparation**: 
   - Converts various document formats (including PDFs, Word documents, emails, and mobile messages) to plaintext or markdown.
   - Applies OCR to scanned documents, with fallback to GPT-4 Vision API for difficult cases.

3. **Document Analysis**: 
   - Splits documents into manageable chunks.
   - Analyzes each chunk for relevance to discovery goals.
   - Extracts key information and generates summaries.
   - Calculates multi-faceted importance scores.

4. **Data Management**: 
   - Stores processed documents and metadata in a SQLite database.
   - Enables full-text search and efficient retrieval of document information.

5. **Dossier Compilation**: 
   - Compiles relevant document summaries into comprehensive dossiers.
   - Generates separate dossiers for high and low importance documents.

6. **Incremental Processing**: 
   - Tracks processed files to efficiently handle updates and new documents.

## Detailed Functionality

### 1. Configuration Generation

- Interprets free-form user input about case details and discovery goals using AI.
- Generates a structured JSON configuration with:
  - Case name
  - Discovery goals (descriptions, keywords, importance ratings)
  - Entities of interest
  - Minimum importance score threshold

### 2. Document Preparation

- Processes diverse formats: PDF, DOC(X), TXT, HTML, PST, EML, mobile message databases.
- Converts documents to plaintext or markdown for uniform processing.
- Applies Tesseract OCR for scanned documents, with GPT-4 Vision API as a fallback.
- Implements specialized handling for email archives and mobile messages.

### 3. Document Analysis

#### a. Chunking
- Splits documents into manageable chunks.
- Uses sophisticated sentence splitting to maintain context and coherence.

#### b. Relevance Assessment
- Analyzes chunks for relevance to discovery goals.
- Identifies key entities and keywords.
- Calculates relevance scores based on goal importance and content matching.

#### c. Information Extraction
- Extracts relevant quotes and passages.
- Identifies document metadata (type, date, author, recipient).
- Generates concise summaries of relevant content.

#### d. Importance Scoring
- Calculates sub-scores for:
  - Relevance to discovery goals
  - Keyword density
  - Entity mentions
  - Temporal relevance
  - Document credibility
  - Information uniqueness
- Computes a weighted average for an overall importance score.

### 4. Data Management

- Creates and populates a SQLite database with:
  - Document content and metadata
  - Case information
  - Discovery goals and keywords
  - Entity relationships
- Implements full-text search for efficient information retrieval.

### 5. Dossier Compilation

- Aggregates processed document information.
- Organizes documents by importance score.
- Generates structured markdown dossiers with:
  - Table of contents
  - Document summaries
  - Key extracts
  - Relevance explanations
  - Importance scores and breakdowns

### 6. Incremental Processing

- Maintains a record of processed files and their hash values.
- Processes only new or modified files in subsequent runs.
- Enables efficient updating of the database and dossiers.

### 7. Performance Optimization

- Implements parallel processing for document analysis using multiprocessing.
- Uses asyncio for efficient I/O operations.
- Applies rate limiting and retry logic for API calls.

### 8. Specialized Functionality

- Processes Enron email corpus for testing and demonstration purposes.
- Provides options for using different AI providers (OpenAI, Claude) or local LLMs.
- Implements robust error handling and comprehensive logging for troubleshooting and auditing.

---

## More Implementation Details and Rationale

### 1. Configuration Generation

- Uses AI models to interpret free-form user input about case details and discovery goals.
- Generates a structured JSON configuration file with case name, discovery goals, keywords, entities of interest, and importance thresholds.
- Implements multi-step validation to ensure configuration validity.
- Allows manual override with custom JSON configuration files.

Rationale: Balances user-friendly input with structured, machine-readable output for flexible case setup.

### 2. Document Preparation

- Processes diverse formats (PDF, DOC, DOCX, TXT, HTML, PST, EML, mobile message databases) using 'textract' and custom parsers.
- Converts documents to plaintext or markdown for uniform processing.
- Implements OCR using Tesseract, with GPT-4 Vision API as a fallback for difficult documents.
- Uses 'Magika' for accurate MIME type detection.
- Handles Outlook PST files and mobile device message databases.

Rationale: Ensures comprehensive document coverage and consistent processing across various formats, expanding the tool's applicability.

### 3. Document Analysis

- Implements intelligent document chunking with context-aware splitting.
- Uses AI-powered language models for relevance assessment, information extraction, and importance scoring.
- Calculates multi-faceted importance scores considering relevance, keyword density, entity mentions, temporal relevance, credibility, and uniqueness.

Rationale: Enables thorough and nuanced document analysis, balancing depth of analysis with processing efficiency.

### 4. Data Management

- Integrates SQLite database for efficient document and metadata storage.
- Implements full-text search capabilities for quick information retrieval.
- Stores case information, discovery goals, and entity relationships.

Rationale: Enhances data organization, speeds up information retrieval, and enables complex querying capabilities.

### 5. Dossier Compilation

- Aggregates processed information into structured markdown dossiers.
- Generates separate high-importance and low-importance dossiers.
- Includes document summaries, key extracts, relevance explanations, and importance scores.

Rationale: Provides easily navigable, comprehensive output for legal professionals to quickly access relevant information.

### 6. Incremental Processing

- Tracks processed files using hash values for efficient updates.
- Processes only new or modified files in subsequent runs.
- Implements file change detection for smart reprocessing.

Rationale: Optimizes processing time and resources for ongoing cases with large document volumes.

### 7. Performance Optimization

- Uses multiprocessing for parallel document analysis.
- Implements asyncio for efficient I/O operations.
- Applies rate limiting and retry logic with exponential backoff for API calls.

Rationale: Enables efficient processing of large document sets while managing API usage responsibly.

### 8. AI Model Flexibility

- Supports multiple AI providers (OpenAI, Anthropic's Claude) and local LLMs.
- Implements provider-specific API calls with error handling.
- Allows easy switching between providers via configuration.

Rationale: Offers flexibility in model selection based on privacy, cost, or capability requirements.

### 9. Specialized Functionality

- Includes automated processing of the Enron email corpus for testing and demonstration.
- Implements robust error handling and comprehensive logging.
- Provides options for suppressing headers and page numbers in processed documents.

Rationale: Enhances the tool's utility for testing, troubleshooting, and handling specific document processing challenges.

## Requirements

- Python 3.12+
- Tesseract OCR engine: *Used for optical character recognition on scanned documents and images.*
- **aiolimiter**: *Implements rate limiting for asynchronous API calls.*
- **aiofiles**: *Provides asynchronous file I/O operations.*
- **anthropic**: *Allows integration with Anthropic's Claude AI model.*
- **backoff**: *Implements exponential backoff for API retries.*
- **filelock**: *Used for file locking to prevent concurrent access issues.*
- **httpx**: *Performs HTTP requests, particularly for downloading files.*
- **libpff-python** (pypff): *Processes Outlook PST files.*
- **llama-cpp-python**: *Enables the use of local LLM models.*
- **magika**: *Detects MIME types of input files.*
- **numpy**: *Used for numerical operations, particularly in image processing.*
- **nvgpu** (optional): *Checks for GPU availability for local LLM processing.*
- **openai**: *Integrates with OpenAI's API for AI-powered text processing.*
- **opencv-python-headless**: *Performs image preprocessing for OCR.*
- **pdf2image**: *Converts PDF pages to images for OCR processing.*
- **picologging**: *Provides efficient logging functionality.*
- **pillow** (PIL): *Handles image processing tasks.*
- **PyPDF2**: *Extracts text from PDF files.*
- **pytesseract**: *Interfaces with Tesseract OCR engine.*
- **python-decouple**: *Manages configuration and environment variables.*
- **tenacity**: *Implements retry logic for API calls.*
- **textract**: *Extracts text from various document formats.*
- **tiktoken**: *Estimates token counts for OpenAI models.*
- **tqdm**: *Displays progress bars for long-running operations.*
- **transformers**: *Used for tokenization with certain AI models.*

Optional requirements:
- OpenAI API key: *Required if using OpenAI's models for text processing.*
- Anthropic API key: *Required if using Anthropic's Claude model.*
- GGUF-compatible model: *Required for local LLM support.*

Note: Additional system libraries may be required for textract and its dependencies, such as antiword, unrtf, and poppler-utils.

## Installation

### 1. Install system dependencies

First, install the required system libraries:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git \
libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr \
flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig redis-server \
libpoppler-cpp-dev pkg-config
```

### 2. Install Pyenv and Python 3.12 (if needed)

```bash
if ! command -v pyenv &> /dev/null; then
    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
    source ~/.bashrc
fi
cd ~/.pyenv && git pull && cd -
pyenv install 3.12
```

### 3. Set up the project

```bash
git clone https://github.com/Dicklesworthstone/llm_aided_legal_discovery_bot.git
cd llm_aided_legal_discovery_bot
pyenv local 3.12
python -m venv venv
source venv/bin/activate
python -m pip install 'pip<24.1' # Pin pip version to avoid issues with textract
python -m pip install wheel
python -m pip install --upgrade setuptools wheel
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root directory and add your API keys:

```
USE_LOCAL_LLM=False
API_PROVIDER=OPENAI
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Usage

1. Place your source documents in the `folder_of_source_documents__original_format` directory.

2. Run the main script:
   ```
   python main.py
   ```

3. Follow the prompts to input your case details and discovery goals.

4. The script will process the documents and generate the dossiers.

The script will generate several output files, including:
- `[case_name]_dossier.md`: Contains summaries and analyses of highly relevant documents.
- `[case_name]_low_importance_dossier.md`: Contains information from less relevant documents for completeness.
- `[case_name].log`: Detailed processing information.

## Configuration

- Modify the `USE_OVERRIDE_DISCOVERY_CONFIG_JSON_FILE` flag in `main.py` to use a pre-existing configuration file.
- Adjust the `OVERRIDE_CONFIG_FILE_PATH` if using a custom configuration file.
- Fine-tune parameters such as chunk sizes, importance score thresholds, and concurrency limits in the main script.

## Output

The script generates two main output files:

1. `[case_name]_dossier.md`: Contains summaries and analyses of highly relevant documents.
2. `[case_name]_low_importance_dossier.md`: Contains information from less relevant documents for completeness.

Additionally, a log file `[case_name].log` is generated with detailed processing information.

## Performance Considerations

- Processing time scales with the number and size of input documents.
- API rate limits may affect processing speed when using cloud-based LLMs.

## Limitations and Future Improvements

- Currently limited to text-based analysis; future versions could incorporate image and audio analysis.
- Accuracy depends on the quality of the AI models used; regular updates to models can improve performance.
- Could benefit from a user interface for easier configuration and result visualization.

---

## In-Depth Example: Enron Documents and Emails

### Overview of the Enron Case

The Enron scandal of 2001 remains a landmark case in corporate fraud, significantly impacting corporate governance and financial regulation. Key aspects include:

- Complex fraudulent accounting using mark-to-market practices and special purpose entities (SPEs)
- Concealment of billions in debt through off-books partnerships
- Inflated profits and stock prices
- Insider trading by executives
- Document destruction by Enron's auditor, Arthur Andersen

Consequences included massive shareholder losses, executive convictions, Arthur Andersen's dissolution, and the Sarbanes-Oxley Act of 2002.

### Significance for Legal Discovery

The Enron case has become a gold standard for testing and developing legal discovery tools, forensic accounting methods, and natural language processing systems. This is due to several factors:

1. **Volume and Variety of Data**: The Enron email corpus, released as part of the investigation, contains over 500,000 emails from 150 employees. This vast dataset provides a realistic scenario for testing large-scale document processing and analysis tools.

2. **Real-world Complexity**: The documents span a wide range of topics, from mundane office communications to discussions of complex financial transactions, offering a true-to-life challenge for content analysis and relevance determination.

3. **Known Outcomes**: With the benefit of hindsight and thorough investigations, we know what kind of evidence exists in these documents, making it possible to evaluate the effectiveness of discovery tools.

4. **Linguistic Diversity**: The emails capture natural language use in a corporate setting, including formal and informal communications, technical jargon, and attempts at obfuscation.

5. **Temporal Aspect**: The dataset spans several years, allowing for analysis of how communications and practices evolved over time, especially as the company approached its collapse.
   


### Implementation in Our Legal Discovery Automation Tool

The Enron case thus provides an ideal testbed for this project; we can get and process all the data and run the tool on the processed data and see empirically whether the system is able to find the relevant documents/emails that would be useful in a civil or criminal case. 

Our tool leverages the Enron case for testing and demonstration:

1. **Automated Data Collection**:
   ```python
   async def download_and_extract_enron_emails_dataset(url: str, destination_folder: str):
       # Downloads and extracts the Enron email dataset
   ```
   - Asynchronously downloads the Enron email dataset.
   - Handles large file downloads with progress tracking using `tqdm`.
   - Extracts the dataset, organizing it into a usable structure.

2. **Email Corpus Processing**:
   ```python
   async def process_extracted_enron_emails(maildir_path: str, converted_source_dir: str):
       # Processes the extracted Enron emails
   ```
   - Parses individual emails, extracting metadata and content.
   - Converts emails to a standardized format for analysis.
   - Implements concurrent processing for efficiency.

3. **Specialized Email Parsing**:
   ```python
   async def parse_enron_email_async(file_path: str) -> Dict[str, Any]:
       # Parses individual Enron emails
   ```
   - Extracts key email fields (From, To, Subject, Date, Cc, Bcc).
   - Handles Enron-specific metadata (X-Folder, X-Origin, X-FileName).
   - Processes email body, applying minimal cleaning.

4. **Document Analysis Pipeline**:
   - Applies OCR to scanned PDFs using Tesseract, with GPT-4 Vision API as a fallback for difficult documents.
   - Utilizes LLMs to analyze content, identify entities, and extract relevant information.
   - Assigns importance scores based on relevance to specified discovery goals.

5. **Dossier Compilation**:
   ```python
   def compile_dossier_section(processed_chunks: List[Dict[str, Any]], document_id: str, file_path: str, discovery_params: Dict[str, Any]) -> Dict[str, Any]:
       # Compiles processed chunks into a cohesive dossier section
   ```
   - Aggregates processed information into structured dossiers.
   - Separates high-importance and low-importance documents.
   - Generates summaries, extracts key information, and provides relevance explanations.

6. **Tailored Discovery Goals**:
   - The `USER_FREEFORM_TEXT_GOAL_INPUT` variable contains specific discovery goals related to the Enron case, such as:
     - Uncovering evidence of financial misreporting and fraudulent accounting practices
     - Identifying communications about off-book entities and debt concealment
     - Locating discussions about market manipulation and insider trading
     - Finding attempts to influence auditors or hide information
   - These goals closely mirror the actual issues in the Enron case, providing a realistic scenario for legal discovery.

7. **Performance Optimization**:
   - Implements parallel processing using `multiprocessing`.
   - Uses `asyncio` for efficient I/O operations, crucial for handling large email datasets.

### Why It's an Excellent Test of the System

1. **Scale and Complexity**: The Enron dataset tests the system's ability to handle a large volume of diverse documents efficiently.

2. **Pattern Recognition**: It challenges the tool to identify subtle patterns of fraudulent behavior across numerous documents and communications.

3. **Contextual Understanding**: The system must understand complex financial concepts and corporate jargon to accurately assess document relevance.

4. **Temporal Analysis**: The tool can demonstrate its ability to track the evolution of issues over time, crucial in understanding how the fraud developed.

5. **Entity Relationship Mapping**: By correctly identifying key players and their roles, the system showcases its capability in building a coherent narrative from fragmented information.

6. **Accuracy and Recall**: With known outcomes, we can evaluate how well the system uncovers critical pieces of evidence that were instrumental in the actual Enron investigation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
