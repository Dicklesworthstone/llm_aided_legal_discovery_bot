# LLM-Aided Legal Discovery Automation

## Table of Contents

## Introduction

 LLM-Aided Legal Discovery Automation is a sophisticated tool designed to automate much of the process of legal discovery by leveraging advanced AI models. This project addresses the common challenges faced by legal professionals when dealing with large volumes of documents in cases such as corporate litigation, intellectual property disputes, or regulatory investigations.

Traditional legal discovery processes are often overwhelming, requiring legal teams to manually sift through thousands or even millions of documents to find relevant information. This manual approach is not only time-consuming and labor-intensive but also prone to human error, potentially missing crucial evidence or insights.

Our Legal Discovery Automation tool transforms this process by:

1. **Simplifying User Input**: Users can easily specify their discovery goals through a structured template or by explaining the key issues and what they are trying to establish or argue in their case (in which case the tool will generate a structured JSON configuration file for you automatically).

2. **Handling Diverse Document Types**: The system processes a wide range of document formats, including scanned PDFs, native digital PDFs, .doc, .docx, .txt, .html files, and even Outlook email archives.

3. **Intelligent Document Analysis**: Each document is analyzed for relevance to the specified discovery goals, with key information extracted and summarized.

4. **Comprehensive Dossier Generation**: The tool produces a detailed dossier of relevant information, including where to find it, why it's relevant, and an estimated importance score.

5. **Flexible Processing**: The system can handle both unstructured text and markdown-formatted documents, making it versatile for various input types.

6. **Maintaining Transparency**: A detailed log of all operations is kept, ensuring the discovery process is auditable and reproducible.

By automating the initial stages of document review and analysis, this tool allows legal professionals to focus their expertise on high-level strategy and decision-making, rather than getting bogged down in the initial document review process. It not only saves time and reduces costs but also enhances the quality and comprehensiveness of the legal discovery process.

The Legal Discovery Automation tool is designed with flexibility and scalability in mind, capable of handling cases of all sizes and complexities. Whether you're dealing with a small internal investigation or a large-scale multi-national litigation, this tool can significantly streamline your workflow and improve your team's efficiency. This tool is particularly useful for law firms, corporate legal departments, and legal technology companies dealing with cases involving large volumes of documents, such as corporate litigation, intellectual property disputes, or regulatory investigations.

## Key Features

- Automated document processing and conversion
- AI-powered relevance assessment
- Intelligent extraction of key information
- Customizable discovery goals and parameters
- Importance scoring for document prioritization
- Generation of comprehensive dossiers
- Support for various document formats (including PDFs requiring OCR)
- Parallel processing for improved performance
- Incremental processing with tracking of already processed files

## How It Works: High-Level Overview

1. **Configuration**: The system generates or loads a configuration file based on user input, defining discovery goals, keywords, and entities of interest.

2. **Document Preparation**: Source documents are converted to plaintext for uniform processing.

3. **Document Analysis**: Each document is processed in parallel:
   - The content is split into manageable chunks.
   - Each chunk is analyzed for relevance to the discovery goals.
   - Key information is extracted and summarized.
   - An importance score is calculated based on relevance and content.

4. **Dossier Compilation**: Relevant document summaries are compiled into a comprehensive dossier, organized by importance.

5. **Output Generation**: The system produces two main outputs:
   - A primary dossier containing highly relevant document summaries.
   - A secondary dossier with lower importance documents for completeness.

## Detailed Functionality

### 1. Configuration Generation

- Uses AI to interpret free-form user input about case details and discovery goals.
- Generates a structured JSON configuration file containing:
  - Case name
  - Discovery goals with descriptions, keywords, and importance ratings
  - Entities of interest
  - Minimum importance score threshold

### 2. Document Preparation

- Supports various input formats (PDF, DOC, DOCX, TXT, etc.)
- Converts all documents to plaintext for uniform processing
- Implements OCR for scanned documents or images

### 3. Document Analysis

#### a. Chunking
- Splits documents into manageable chunks for processing
- Implements intelligent splitting to maintain context and coherence

#### b. Relevance Assessment
- Analyzes each chunk for relevance to discovery goals
- Identifies mentions of key entities and keywords
- Calculates a relevance score based on goal importance and content matching

#### c. Information Extraction
- Extracts key quotes and passages relevant to discovery goals
- Identifies document metadata (type, date, author, recipient)
- Generates concise summaries of relevant content

#### d. Importance Scoring
- Calculates sub-scores for various aspects:
  - Relevance to discovery goals
  - Keyword density
  - Entity mentions
  - Temporal relevance
  - Document credibility
  - Information uniqueness
- Computes a weighted average for an overall importance score

### 4. Dossier Compilation

- Aggregates processed document information
- Organizes documents by importance score
- Generates a structured markdown document with:
  - Table of contents
  - Document summaries
  - Key extracts
  - Relevance explanations
  - Importance scores and breakdowns

### 5. Incremental Processing

- Maintains a record of processed files and their hash values
- On subsequent runs, only processes new or modified files
- Allows for efficient updating of the dossier with new information

### 6. Performance Optimization

- Implements parallel processing for document analysis
- Uses asyncio for efficient I/O operations
- Implements rate limiting for API calls to prevent overload

## More Implementation Details and Rationale

### 1. Configuration Generation

- Uses AI-powered language models (OpenAI or Claude) to interpret free-form user input about case details and discovery goals.
- Generates a structured JSON configuration file containing:
  - Case name
  - Discovery goals with descriptions, keywords, and importance ratings
  - Entities of interest
  - Minimum importance score threshold
- Implements a multi-step process to ensure the generated configuration is valid:
  1. Initial JSON generation based on user input
  2. JSON structure validation and correction
  3. Python-based validation of the corrected JSON
  4. Generation of a descriptive file name for the configuration
- Saves the configuration in a dedicated folder for easy access and management

Why: This approach allows for flexible and user-friendly input while ensuring the resulting configuration is structured and valid for use in the discovery process.

### 2. Document Preparation

- Supports various input formats (PDF, DOC, DOCX, TXT, HTML, emails) using the 'textract' library for uniform text extraction.
- Converts all documents to plaintext for consistent processing.
- Implements OCR for scanned documents or images using Tesseract OCR:
  - Converts PDF pages to images
  - Preprocesses images (grayscale conversion, thresholding, dilation) to improve OCR accuracy
  - Applies OCR to each preprocessed image
- Uses the 'Magika' library for accurate MIME type detection of input files.
- Implements custom email parsing for '.eml' files to extract metadata and body content.

Why: Uniform text extraction ensures consistent processing across different file formats, while OCR capability allows for handling scanned documents, greatly expanding the range of processable inputs.

### 3. Document Analysis

#### a. Chunking
- Splits documents into manageable chunks for processing, typically around 2000 characters each.
- Implements intelligent splitting to maintain context and coherence:
  - Attempts to split at sentence boundaries
  - Ensures overlap between chunks to maintain context
- Uses sophisticated sentence splitting that accounts for abbreviations, quotations, and other edge cases.

Why: Chunking allows for parallel processing of large documents and helps manage token limits of AI models. Intelligent splitting preserves document coherence for more accurate analysis.

#### b. Relevance Assessment
- Analyzes each chunk for relevance to discovery goals using AI-powered language models.
- Identifies mentions of key entities and keywords.
- Calculates a relevance score based on goal importance and content matching.
- Uses a structured prompt template to guide the AI in assessing relevance consistently.

Why: This step filters out irrelevant content early in the process, focusing further analysis on potentially important information.

#### c. Information Extraction
- Extracts key quotes and passages relevant to discovery goals.
- Identifies document metadata (type, date, author, recipient).
- Generates concise summaries of relevant content.
- Uses AI to highlight and explain the significance of extracted information.

Why: This provides a condensed view of the most important information in each document, making it easier for legal professionals to quickly grasp key points.

#### d. Importance Scoring
- Calculates sub-scores for various aspects:
  - Relevance to discovery goals
  - Keyword density
  - Entity mentions
  - Temporal relevance
  - Document credibility
  - Information uniqueness
- Computes a weighted average for an overall importance score.
- Provides justifications for each sub-score to explain the rationale behind the scoring.

Why: The multi-faceted scoring system provides a nuanced view of document importance, helping prioritize documents for review.

### 4. Dossier Compilation

- Aggregates processed document information.
- Organizes documents by importance score.
- Generates a structured markdown document with:
  - Table of contents
  - Document summaries
  - Key extracts
  - Relevance explanations
  - Importance scores and breakdowns
- Creates separate dossiers for high-importance and low-importance documents.

Why: This step synthesizes the analyzed information into a coherent, easily navigable document, allowing legal professionals to quickly access the most relevant information.

### 5. Incremental Processing

- Maintains a record of processed files and their hash values.
- On subsequent runs, only processes new or modified files.
- Uses MD5 hashing to efficiently detect file changes.
- Allows for efficient updating of the dossier with new information.

Why: This feature saves time and computational resources by avoiding redundant processing of unchanged documents in large, ongoing cases.

### 6. Performance Optimization

- Implements parallel processing for document analysis using Python's multiprocessing module.
- Uses asyncio for efficient I/O operations, particularly in API calls and file operations.
- Implements rate limiting for API calls to prevent overload and comply with service limits.
- Utilizes a semaphore to control concurrent API requests.
- Implements retry logic with exponential backoff for API calls to handle transient errors.

Why: These optimizations allow the system to process large volumes of documents efficiently, making it scalable for big cases while respecting API rate limits.

### 7. Flexibility in Language Model Selection

- Supports multiple AI providers (OpenAI, Anthropic's Claude) and local LLMs.
- Implements provider-specific API calls with appropriate error handling and retries.
- Allows for easy switching between providers through configuration settings.

Why: This flexibility allows users to choose the most suitable or cost-effective AI provider for their needs, or to use local models for enhanced privacy and offline capability.

### 8. Robust Error Handling and Logging

- Implements comprehensive error handling throughout the codebase.
- Uses Python's logging module to provide detailed logs of the entire process.
- Creates session-specific log files for easy troubleshooting and auditing.

Why: Proper error handling and logging are crucial for maintaining the reliability of the system and for debugging issues in a complex, multi-stage process.

## Requirements

- Python 3.12+
- Tesseract OCR engine
- PDF2Image library
- PyTesseract
- Textract and its dependencies
- OpenAI API (optional)
- Anthropic API (optional)
- Local LLM support (optional, requires compatible GGUF model)

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

## In-Depth Example: Enron Documents and Emails

### Overview of the Enron Case

The Enron scandal, which unfolded in 2001, stands as one of the most notorious corporate fraud cases in American history. Enron Corporation, once hailed as America's most innovative company, collapsed due to widespread accounting fraud and corruption. This case has become a landmark in corporate governance, accounting practices, and regulatory reform.

Key aspects of the Enron scandal include:
- Fraudulent accounting practices using mark-to-market accounting and special purpose entities (SPEs)
- Off-the-books partnerships used to hide billions in debt and toxic assets
- Inflated profits and stock prices
- Insider trading by top executives
- Destruction of documents by Enron's auditor, Arthur Andersen

The fall of Enron led to:
- Billions of dollars in losses for shareholders and employees
- Criminal convictions of top executives, including CEO Jeffrey Skilling and CFO Andrew Fastow
- The dissolution of Arthur Andersen, one of the "Big Five" accounting firms
- The passage of the Sarbanes-Oxley Act of 2002, significantly reforming corporate financial practices and reporting

### Significance for Legal Discovery and Analysis

The Enron case has become a gold standard for testing and developing legal discovery tools, forensic accounting methods, and natural language processing systems. This is due to several factors:

1. **Volume and Variety of Data**: The Enron email corpus, released as part of the investigation, contains over 500,000 emails from 150 employees. This vast dataset provides a realistic scenario for testing large-scale document processing and analysis tools.

2. **Real-world Complexity**: The documents span a wide range of topics, from mundane office communications to discussions of complex financial transactions, offering a true-to-life challenge for content analysis and relevance determination.

3. **Known Outcomes**: With the benefit of hindsight and thorough investigations, we know what kind of evidence exists in these documents, making it possible to evaluate the effectiveness of discovery tools.

4. **Linguistic Diversity**: The emails capture natural language use in a corporate setting, including formal and informal communications, technical jargon, and attempts at obfuscation.

5. **Temporal Aspect**: The dataset spans several years, allowing for analysis of how communications and practices evolved over time, especially as the company approached its collapse.

### Implementation in Our Legal Discovery Automation Tool

Our project leverages the Enron case in several ways to create a robust testing and demonstration environment:

1. **Data Collection**:
   - The script `enron_sample_data_collector_script.py` automatically downloads a comprehensive set of Enron-related documents from the U.S. Department of Justice archive.
   - It fetches PDF exhibits, which include financial statements, internal memos, and other crucial documents related to the case.
   - The script uses asynchronous programming to efficiently download hundreds of PDFs, handling potential network issues and retries.

2. **Enron Email Corpus Processing**:
   - The main script includes functionality to download and process the Enron email dataset.
   - It extracts emails from the downloaded maildir structure, converting them into a format suitable for our analysis pipeline.
   - This process demonstrates the tool's ability to handle diverse document types, including emails with headers and attachments.

3. **Tailored Discovery Goals**:
   - The `USER_FREEFORM_TEXT_GOAL_INPUT` variable contains specific discovery goals related to the Enron case, such as:
     - Uncovering evidence of financial misreporting and fraudulent accounting practices
     - Identifying communications about off-book entities and debt concealment
     - Locating discussions about market manipulation and insider trading
     - Finding attempts to influence auditors or hide information
   - These goals closely mirror the actual issues in the Enron case, providing a realistic scenario for legal discovery.

4. **Document Processing and Analysis**:
   - The tool applies OCR to scanned PDFs, ensuring that all document types, including handwritten notes and complex financial statements, are searchable.
   - It uses sophisticated NLP techniques to analyze the content, identify key entities (like Kenneth Lay, Jeffrey Skilling, and Andrew Fastow), and extract relevant information.
   - The system assigns importance scores to documents based on their relevance to the specified discovery goals, mimicking the prioritization process in a real investigation.

5. **Dossier Compilation**:
   - The final output includes a comprehensive dossier of high-importance documents and a separate dossier for low-importance but potentially relevant documents.
   - This structure reflects how legal teams might organize findings in a complex case like Enron, separating smoking guns from contextual information.

### Why It's an Excellent Test of the System

1. **Scale and Complexity**: The Enron dataset tests the system's ability to handle a large volume of diverse documents efficiently.

2. **Pattern Recognition**: It challenges the tool to identify subtle patterns of fraudulent behavior across numerous documents and communications.

3. **Contextual Understanding**: The system must understand complex financial concepts and corporate jargon to accurately assess document relevance.

4. **Temporal Analysis**: The tool can demonstrate its ability to track the evolution of issues over time, crucial in understanding how the fraud developed.

5. **Entity Relationship Mapping**: By correctly identifying key players and their roles, the system showcases its capability in building a coherent narrative from fragmented information.

6. **Accuracy and Recall**: With known outcomes, we can evaluate how well the system uncovers critical pieces of evidence that were instrumental in the actual Enron investigation.

### Educational and Developmental Value

Incorporating the Enron case into our legal discovery automation tool serves multiple purposes:

1. **Realistic Training Data**: It provides developers and users with a rich, real-world dataset to refine and test the system's capabilities.

2. **Benchmark for Improvements**: As we enhance the tool's algorithms and features, we can consistently test against this well-understood case to measure progress.

3. **Demonstration of Capabilities**: For potential users or stakeholders, the Enron example vividly illustrates the tool's power in handling complex, high-stakes legal discovery scenarios.

4. **Ethical Considerations**: It serves as a reminder of the importance of corporate ethics and the role technology can play in uncovering wrongdoing.

5. **Interdisciplinary Learning**: The Enron case touches on law, finance, ethics, and data analysis, making our tool a valuable resource for educational purposes across multiple disciplines.

By integrating this infamous case into our legal discovery automation tool, we not only create a powerful testing and demonstration platform but also contribute to the ongoing study and understanding of one of the most significant corporate scandals in modern history. This implementation showcases our tool's capability to handle real-world, complex legal discovery challenges, positioning it as a valuable asset in the fields of law, finance, and corporate governance.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
