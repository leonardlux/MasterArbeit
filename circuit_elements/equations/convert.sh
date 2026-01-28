#bin/bash

for pdf in *.pdf; do
	pdf2svg "$pdf" "${pdf%.pdf}.svg"
done
