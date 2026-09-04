from pypdf import PdfReader


def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()
        pages.append(text)

    return pages


if __name__ == "__main__":
    pdf_pages = load_pdf("data/Attention_Is_All_You_Need.pdf")

    print(f"Number of pages: {len(pdf_pages)}")

    print("\nFirst 500 characters:\n")

    print(pdf_pages[0][:500])