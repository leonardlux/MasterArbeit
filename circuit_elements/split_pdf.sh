#bin/bash

for pdf in *.pdf; do
    # Get the number of pages in the PDF
    num_pages=$(pdfinfo "$pdf" | grep Pages | awk '{print $2}')
    
    # Loop through each page of the PDF
    for ((i=1; i<=num_pages; i++)); do
        # Convert each page to SVG
        pdf2svg "$pdf" "${pdf%.pdf}_page$i.svg" $i
    done
done

