TEX_DIR  := paper
MAIN     := main
PDF      := $(TEX_DIR)/$(MAIN).pdf

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX_DIR)/$(MAIN).tex $(TEX_DIR)/definitions.tex $(TEX_DIR)/proof.tex $(TEX_DIR)/references.bib
	cd $(TEX_DIR) && latexmk -pdf -interaction=nonstopmode $(MAIN).tex

clean:
	cd $(TEX_DIR) && latexmk -C $(MAIN).tex
