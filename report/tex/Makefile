TGT = paper.tex

all: $(TGT)
	latexmk -bibtex -shell-escape -xelatex $<

clean:
	latexmk -bibtex -C $(TGT)
