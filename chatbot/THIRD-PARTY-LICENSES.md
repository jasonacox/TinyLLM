# Third-Party Software Licenses

This project uses the following third-party libraries:

## Frontend Libraries (JavaScript/CSS)

### Prism.js
- **License**: MIT License
- **Copyright**: Copyright (c) 2012 Lea Verou
- **Website**: https://prismjs.com/
- **Purpose**: Syntax highlighting for code blocks in the chat interface
- **Files**: 
  - `app/static/prism.min.js`
  - `app/static/prism.min.css`
  - `app/static/prism-python.min.js`
  - `app/static/prism-javascript.min.js`
  - `app/static/prism-markup.min.js`

### marked.js
- **License**: MIT License
- **Copyright**: Copyright (c) 2018+, MarkedJS (https://github.com/markedjs/)
- **Website**: https://marked.js.org/
- **Purpose**: Markdown parsing and rendering in the chat interface
- **Files**: `app/static/marked.min.js`

### Socket.IO
- **License**: MIT License
- **Copyright**: Copyright (c) 2014-present Automattic <dev@cloudup.com>
- **Website**: https://socket.io/
- **Purpose**: Real-time bidirectional communication between client and server
- **Files**: `app/static/socket.io.js`

## Backend Libraries (Python)

### FastAPI
- **License**: MIT License
- **Copyright**: Copyright (c) 2018 Sebastián Ramírez
- **Website**: https://fastapi.tiangolo.com/
- **Purpose**: Modern web framework for building APIs

### Uvicorn
- **License**: BSD License
- **Website**: https://www.uvicorn.org/
- **Purpose**: ASGI web server implementation

### Python-SocketIO
- **License**: MIT License
- **Website**: https://python-socketio.readthedocs.io/
- **Purpose**: Server-side Socket.IO implementation

### OpenAI Python Client
- **License**: MIT License
- **Website**: https://github.com/openai/openai-python
- **Purpose**: OpenAI API client for LLM interaction

### Beautiful Soup 4 (bs4)
- **License**: MIT License
- **Website**: https://www.crummy.com/software/BeautifulSoup/
- **Purpose**: HTML/XML parsing for web scraping

### PyPDF
- **License**: BSD License
- **Website**: https://github.com/py-pdf/pypdf
- **Purpose**: PDF file reading and text extraction

### Pillow & pillow-heif
- **License**: HPND License (Historical Permission Notice and Disclaimer)
- **Website**: https://python-pillow.org/
- **Purpose**: Image processing and HEIF format support

### Weaviate Client
- **License**: BSD-3-Clause License
- **Website**: https://weaviate.io/
- **Purpose**: Vector database client for RAG functionality

### ReportLab
- **License**: BSD License
- **Website**: https://www.reportlab.com/opensource/
- **Purpose**: PDF document generation

### python-docx
- **License**: MIT License
- **Website**: https://python-docx.readthedocs.io/
- **Purpose**: Microsoft Word document generation

### python-pptx
- **License**: MIT License
- **Website**: https://python-pptx.readthedocs.io/
- **Purpose**: Microsoft PowerPoint document generation

### openpyxl
- **License**: MIT License
- **Website**: https://openpyxl.readthedocs.io/
- **Purpose**: Excel spreadsheet generation

### pypandoc
- **License**: MIT License
- **Website**: https://github.com/bebraw/pypandoc
- **Purpose**: Document format conversion

### aiohttp
- **License**: Apache License 2.0
- **Website**: https://docs.aiohttp.org/
- **Purpose**: Async HTTP client/server framework

### Requests
- **License**: Apache License 2.0
- **Website**: https://requests.readthedocs.io/
- **Purpose**: HTTP library for API calls

### lxml
- **License**: BSD License
- **Website**: https://lxml.de/
- **Purpose**: XML and HTML processing

### Jinja2
- **License**: BSD License
- **Website**: https://jinja.palletsprojects.com/
- **Purpose**: Template engine for HTML rendering

### Pydantic
- **License**: MIT License
- **Website**: https://pydantic-docs.helpmanual.io/
- **Purpose**: Data validation using Python type annotations

### python-dotenv
- **License**: BSD License
- **Website**: https://github.com/theskumar/python-dotenv
- **Purpose**: Environment variable management

### pandas
- **License**: BSD License
- **Website**: https://pandas.pydata.org/
- **Purpose**: Data manipulation and analysis

---

## MIT License Text

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Apache License 2.0

Libraries using Apache License 2.0 (aiohttp, requests) are licensed under terms that allow free use, modification, and distribution. Full license text available at: https://www.apache.org/licenses/LICENSE-2.0

## BSD License

Libraries using BSD License (Uvicorn, PyPDF, ReportLab, Jinja2, python-dotenv, pandas, lxml, Weaviate Client) are licensed under permissive terms similar to MIT. Each has slight variations but all allow free use, modification, and distribution with attribution.

---

All third-party libraries are used in accordance with their respective licenses.
