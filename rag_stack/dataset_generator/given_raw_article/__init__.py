import logging
from typing import List, Callable

from langchain_community.document_loaders import (
	PDFMinerLoader,
	PDFPlumberLoader,
	PyPDFium2Loader,
	PyPDFLoader,
	PyMuPDFLoader,
	UnstructuredPDFLoader,
	CSVLoader,
	JSONLoader,
	UnstructuredMarkdownLoader,
	BSHTMLLoader,
	UnstructuredXMLLoader,
	DirectoryLoader,
)
from langchain_unstructured import UnstructuredLoader

# langchain_upstage is an OPTIONAL document-parse backend. Recent langchain_upstage
# releases import the removed ``langchain_core.pydantic_v1``, which hard-crashes this
# whole module on import. Make it optional so the (far more common) non-Upstage parse
# backends keep working; ``upstagedocumentparse`` is registered only when the package
# imports cleanly.
try:
	from langchain_upstage import UpstageLayoutAnalysisLoader
except Exception:  # noqa: BLE001 — any import-time failure ⇒ backend unavailable
	UpstageLayoutAnalysisLoader = None

logger = logging.getLogger("RAG-Stack")

parse_modules = {
	# PDF
	"pdfminer": PDFMinerLoader,
	"pdfplumber": PDFPlumberLoader,
	"pypdfium2": PyPDFium2Loader,
	"pypdf": PyPDFLoader,
	"pymupdf": PyMuPDFLoader,
	"unstructuredpdf": UnstructuredPDFLoader,
	# Common File Types
	"csv": CSVLoader,
	"json": JSONLoader,
	"unstructuredmarkdown": UnstructuredMarkdownLoader,
	"bshtml": BSHTMLLoader,
	"unstructuredxml": UnstructuredXMLLoader,
	"directory": DirectoryLoader,
	"unstructured": UnstructuredLoader,
}
if UpstageLayoutAnalysisLoader is not None:
	parse_modules["upstagedocumentparse"] = UpstageLayoutAnalysisLoader

# Re-export from canonical location in static_rag_evaluator
from rag_stack_evaluator.static_rag_evaluator.chunk._registry import chunk_modules, sentence_splitter_modules
