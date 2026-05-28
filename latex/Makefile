# Makefile untuk compile LaTeX document
# Usage: make          (compile PDF)
#        make clean    (hapus file temporary)
#        make view     (open PDF)

.PHONY: all clean view help

LATEX    := pdflatex
BIBTEX   := bibtex
MAIN     := main
OUTPUT   := $(MAIN).pdf

# Default target
all: $(OUTPUT)

# Generate PDF
$(OUTPUT): $(MAIN).tex preamble.tex bib/custom.bib sections/*.tex
	$(LATEX) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

# Clean temporary files
clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log
	rm -f $(MAIN).out $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk
	rm -f *~

# Remove all generated files including PDF
distclean: clean
	rm -f $(OUTPUT)

# View the PDF (works on Windows with start command)
view: $(OUTPUT)
	start $(OUTPUT)

# Help
help:
	@echo "Available targets:"
	@echo "  make all       - Compile PDF (default)"
	@echo "  make clean     - Remove temporary files"
	@echo "  make distclean - Remove all generated files"
	@echo "  make view      - Open PDF viewer"
	@echo "  make help      - Show this message"
