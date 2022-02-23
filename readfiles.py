import os

def readfiles():
    # Folder Path
    path = "/Users/alfredosuarez/Desktop/Files/Sentiment-Analysis/Datasets/Texts"

    paragraphs = []

    # Change the directory
    os.chdir(path)

    # Read text File
    def read_text_file(file_path, lst):
        with open(file_path, 'r', encoding="iso-8859-1") as f:
            paragraph = f.read()
            paragraph_list = paragraph.split()
            alnum_paragraph = []
            for word in paragraph_list:
                is_word_alphanumeric = word.isalpha()
                if is_word_alphanumeric:
                    alnum_paragraph.append(word.lower())
            cleaned_paragraph = " ".join(alnum_paragraph)
            lst.append(cleaned_paragraph)


    # Iterate through all file
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}/{file}"
            # call read text file function
            read_text_file(file_path, paragraphs)

    return paragraphs
