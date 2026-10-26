# Build the arXiv-style PDF from the markdown writeup.
#   make pdf    -> off_the_prior.pdf (LaTeX source kept in build/)
SRC = off_the_prior.md
.PHONY: pdf clean
pdf: build/off_the_prior.tex
	cd build && latexmk -xelatex -interaction=nonstopmode -halt-on-error off_the_prior.tex >/dev/null && cp off_the_prior.pdf ..
build/off_the_prior.tex: $(SRC) template.tex
	mkdir -p build
	cp -r figures build/ 2>/dev/null || true
	pandoc $(SRC) --standalone --template=template.tex -o build/off_the_prior.tex
clean:
	rm -rf build off_the_prior.pdf
