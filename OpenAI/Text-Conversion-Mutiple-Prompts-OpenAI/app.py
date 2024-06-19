
# 1. Program will take as input text with markdown from ARGV. Note these will be paragraps or larger (not single sentences).
# 2. We will identify each sentence.
# 3. We will mark on the original text the location of each sentence. We do this so at the last step of the program, we know where to put the transformed sentence back into the original markdown text.
# 4. We perform coreference resololution on the text.
# 5. We replace into the original text the sentences which werre changed by the correference resolution, in place of the old ones.
## Debug: We create a csv showing original sentene, new sentence.
# 6. We identify sentences in passive voice and conver them to active voice, then put the back in the original text.
## Debug: We create a csv showing original sentene, new sentence.
# 7. We identify sentences that contain the modal verbs (can, could, may, might, shall, will, would, must) and transform them to non modal sentences, then replace the old sentences.
## Debug: We create a csv showing original sentene, new sentence.
# 8. We identify sentences that start with "if" and transform them so they don't start with if.
## Debug: We create a csv showing original sentene, new sentence.
# 9. We identify sentences that start with "there is", "there are" and transform them so they don't start with "there is", "there are".
## Debug: We create a csv showing original sentene, new sentence.
# 10. We Generate create a new .md file with based on the old md, but with all the sentences that were modified in the steps.

import os
import re
import sys
import nltk
import spacy
import pandas as pd

import openai
from openai import OpenAI

import markdown2
from markdown import markdown
from bs4 import BeautifulSoup

debug_folder = "debug"
# Ensure you have the necessary NLTK data files
nltk.download('punkt')

def open_file(file_name):

    try:
        # Read the content of the file
        with open(file_name, 'r', encoding='utf-8') as file:
            markdown_text = file.read()
            return markdown_text
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        return
    except Exception as e:
        print(f"Error: {e}")
        return
    
def create_debug_csv_file(input_list,output_list,column_name, file_name_csv):
    data = {
        'Original Sentence': input_list,
        column_name : output_list
    }

    df = pd.DataFrame(data)
    output_csv = os.path.join(debug_folder, file_name_csv)
    df.to_csv(output_csv, index=False)
    print(f"CSV file created: {output_csv}")

def create_debug_txt_file(text, file_name):
    file_path = os.path.join(debug_folder, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"Text file created: {file_path}")


def print_line_differences(text1, text2,column_name,file_name_csv):
    """Compares two blocks of text line by line and prints the differing lines."""
    # Split the text blocks into lines
    text1_lines = text1.splitlines()
    text2_lines = text2.splitlines()
    
    # Determine the maximum number of lines between the two texts
    max_lines = max(len(text1_lines), len(text2_lines))

    # Iterate through each line index up to the maximum number of lines
    input_sentenses =  []
    output_sentenses = []
    for i in range(max_lines):
        line1 = text1_lines[i] if i < len(text1_lines) else ""
        line2 = text2_lines[i] if i < len(text2_lines) else ""

        # Compare the lines; if they differ, print the line number and the differing lines
        if line1 != line2:
            input_sentenses.append(line1)
            output_sentenses.append(line2)
   
    create_debug_csv_file(input_sentenses,output_sentenses, column_name,file_name_csv)

def markdown_line_to_text(markdown_line: str) -> str:
    """Converts a single line of markdown text to plain text."""
    # Convert markdown to HTML
    html = markdown2.markdown(markdown_line)
    # Use BeautifulSoup to extract text from HTML
    soup = BeautifulSoup(html, features="html.parser")
    return soup.get_text()

def markdown_to_text_preserve_lines(markdown_text: str) -> str:
    """Converts multiline markdown text to plain text, preserving newlines."""
    lines = markdown_text.splitlines()
    plain_text_lines = [markdown_line_to_text(line) for line in lines]
    return ''.join(plain_text_lines)


def split_into_sentences(text: str):
    """Splits text into sentences using NLTK's sentence tokenizer."""
    return nltk.sent_tokenize(text)

def load_nlp_model(model_name):
    """Load the spaCy NLP model."""
    return spacy.load(model_name)


def replace_coreferences(text, coref_clusters):
    """
    Replace coreferences in the text with their main mention.

    Args:
        text (str): The original text.
        coref_clusters (dict): A dictionary of coreference clusters.

    Returns:
        str: The text with coreferences replaced.
    """
    for cluster in coref_clusters.values():
        cluster = list(cluster)
        main_mention = cluster[0]
        for coref in cluster[1:]:
            text = text.replace(f" {coref} ", f" {main_mention} ")
    return text

def replace_coreferences_text(text):
    # Constants
    COREF_MODEL_NAME = "en_coreference_web_trf"
    # Load the coreference model and process the text
    nlp_coref = load_nlp_model(COREF_MODEL_NAME)

    doc_coref = nlp_coref(text)
    #print(doc_coref.spans)

    # Replace coreferences in the text
    updated_text = replace_coreferences(text, doc_coref.spans)

    return updated_text

# Function to read the CSV file and create a replacement dictionary using pandas
def read_replacements_from_csv(csv_file):
    csv_file = os.path.join(debug_folder, csv_file)
    df = pd.read_csv(csv_file)
    return dict(zip(df.iloc[:, 1], df.iloc[:, 0]))

# Function to replace text in a file based on the replacement dictionary
def replace_text_in_file(input_txt_file, output_txt_file, replacements):
    input_file = os.path.join(debug_folder, input_txt_file)
    with open(input_file, mode='r', encoding='utf-8') as file:
        content = file.read()

    for old_text, new_text in replacements.items():
        content = content.replace(old_text, new_text)
    output_file = os.path.join(debug_folder, output_txt_file)
    with open(output_file, mode='w', encoding='utf-8') as file:
        file.write(content)
    return content




# Function to detect passive voice
def is_passive(sentence):
    # Load the spaCy model for English
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == "nsubjpass" or token.dep_ == "auxpass":
            return True
    return False

# Function to convert passive voice to active voice
def convert_to_active_voice(sentence):
    prompt = f"Convert the sentence from passive voice to only active voice.\nPassive Voice -> Active Voice\n{sentence} ->"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Convert the sentence from passive voice to only active voice."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )
    return (response.choices[0].message.content)


def get_text_msrkdown(markeddown_contnet):
    print(type(markeddown_contnet))
    html = markdown(markeddown_contnet)

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Extract normal text into another list
    text = []
    for paragraph in soup.find_all('p'):
        # Remove bold tags from the paragraph text
        for bold in paragraph.find_all(['strong', 'b']):
            bold.extract()
        text.append(paragraph.get_text().strip())
    return text

def find_modal_verbs(sentence):
    # Define a list of modal verbs
    modal_verbs = ["can", "could", "may", "might", "shall", "will", "would", "must", "should"]
    
    # Create a regex pattern to match any of the modal verbs
    pattern = r'\b(' + '|'.join(modal_verbs) + r')\b'
    found_modals = re.findall(pattern, sentence, re.IGNORECASE)
    # Return a list of unique modal verbs found (case insensitive)
    return list(set(found_modals))
    
def convert_non_modal_verb_sentence(sentence, modal_verb):
    prompt = f"Clarify the sentence by removing the modal verb '{modal_verb}' from the sentence, and possibly rearranging, adding, modifying, or deleting other words:\nSentence: {sentence}"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Clarify the sentence by removing the modal verb, and possibly rearranging, adding, modifying, or deleting other words."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )
    
    return (response.choices[0].message.content)

# Function to identify and process modal verbs in a sentence
can_input_list = []
can_output_list = []

could_input_list = []
could_output_list = []

may_input_list = []
may_output_list = []

might_input_list = []
might_output_list = []

shall_input_list = []
shall_output_list = []

will_input_list = []
will_output_list = []

would_input_list = []
would_output_list = []

must_input_list = []
must_output_list = []

should_input_list = []
should_output_list = []

def process_sentence(sentence):
    modals = find_modal_verbs(sentence)
    if len(modals)==0:
        return False
    
    if "can" in modals:
        converted_sentence = convert_non_modal_verb_sentence(sentence,"can")
        print(f"Original: '{sentence}' | Without 'can': '{converted_sentence}'")
        can_input_list.append(sentence)
        can_output_list.append(converted_sentence)
        
    elif "could" in modals:
        converted_sentence = convert_non_modal_verb_sentence(sentence,"could")
        print(f"Original: '{sentence}' | Without 'could': '{converted_sentence}'")
        could_input_list.append(sentence)
        could_output_list.append(converted_sentence)
        
    elif "may" in modals:
        converted_sentence = convert_non_modal_verb_sentence(sentence,"may")
        print(f"Original: '{sentence}' | Without 'may': '{converted_sentence}'")
        may_input_list.append(sentence)
        may_output_list.append(converted_sentence)
        
    elif "might" in modals:
        converted_sentence = convert_non_modal_verb_sentence(sentence,"might")
        print(f"Original: '{sentence}' | Without 'might': '{converted_sentence}'")
        might_input_list.append(sentence)
        might_output_list.append(converted_sentence)
        
    elif "shall" in modals:
        converted_sentence = convert_non_modal_verb_sentence(sentence,"shall")
        print(f"Original: '{sentence}' | Without 'shall': '{converted_sentence}'")
        shall_input_list.append(sentence)
        shall_output_list.append(converted_sentence)
        
    elif "will" in modals:
        converted_sentence = convert_non_modal_verb_sentence(sentence,"will")
        print(f"Original: '{sentence}' | Without 'will': '{converted_sentence}'")
        will_input_list.append(sentence)
        will_output_list.append(converted_sentence)
        
    elif "would" in modals:
        converted_sentence = convert_non_modal_verb_sentence(sentence,"would")
        print(f"Original: '{sentence}' | Without 'would': '{converted_sentence}'")
        would_input_list.append(sentence)
        would_output_list.append(converted_sentence)
        
    elif "must" in modals:
        converted_sentence = convert_non_modal_verb_sentence(sentence,"must")
        print(f"Original: '{sentence}' | Without 'must': '{converted_sentence}'")
        must_input_list.append(sentence)
        must_output_list.append(converted_sentence)

    elif "should" in modals:
        converted_sentence = convert_non_modal_verb_sentence(sentence,"should")
        print(f"Original: '{sentence}' | Without 'should': '{converted_sentence}'")
        should_input_list.append(sentence)
        should_output_list.append(converted_sentence)
    return converted_sentence

def check_if_sentences(sentence: str):
    """Check sentences that start with 'if' so they don't start with 'if'."""
    if sentence.lower().startswith("if "):
        return True
    return False

# Function to transform sentences that start with "if"
def transform_if_sentences(sentence):
    prompt = f"Clarify sentence bellow by removing the (if) from the beggining of the sentence, and possibly rearranging, adding, modifying, or deleting other words:\nSentence: {sentence}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Transform sentences that start with 'if'."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )
    return (response.choices[0].message.content)

def check_there_is_sentences(sentence: str):
    """check sentences that start with 'there is' or 'there are'."""
    if sentence.lower().startswith("there is "):
        return True
    return False
    
def check_there_are_sentences(sentence: str):
    """check sentences that start with 'there is' or 'there are'."""
    if sentence.lower().startswith("there are "):
        return True
    return False

# Function to transform sentences that start with "there is" 
def transform_there_is_sentences(sentence):
    prompt = f" Clarify sentence bellow by removing the (there are) from the beggining of the sentence, and possibly rearranging, adding, modifying, or deleting other words:\nSentence: {sentence}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Transform sentences that start with 'there is'."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )
    return (response.choices[0].message.content)

# Function to transform sentences that start with  "there are"
def transform_there_are_sentences(sentence):
    prompt = f"Clarify sentence bellow by removing the (there are) from the beggining of the sentence, and possibly rearranging, adding, modifying, or deleting other words:\nSentence: {sentence}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Transform sentences that start with 'there are'."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )
    return (response.choices[0].message.content)

def main():

    debug_folder="debug"
    # Check if the output directory exists; if not, create it
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)
    
    # 1. Program will take as input text with markdown from ARGV. 
    ## Note these will be paragraps or larger (not single sentences).
    if len(sys.argv) < 2:
        print("Usage: python app.py <input_file>")
        sys.exit(1)

    file_name = sys.argv[1]    
    markdown_text = open_file(file_name)

    # 3. We will mark on the original text the location of each sentence. 
    ## We do this so at the last step of the program, we know where to put the transformed sentence back 
    ## into the original markdown text.
    plain_text = markdown_to_text_preserve_lines(markdown_text)
    create_debug_txt_file(plain_text,"markdown_to_plain_text.txt")
    print_line_differences(markdown_text, plain_text,"Markdown To Plain text","markdown_to_plain_text.csv")

    # 4. We perform coreference resololution on the text.
    # 5. We replace into the original text the sentences which werre changed by the correference resolution, 
    ## in place of the old ones.
    ### Debug: We create a csv showing original sentene, new sentence.
    # Process each sentence individually
    converted_text = replace_coreferences_text(plain_text)
    create_debug_txt_file(converted_text,"coreferences_text.txt")
    print_line_differences(converted_text, plain_text,"Correference Text", "Correference.csv")

    replacements = read_replacements_from_csv("markdown_to_plain_text.csv")

    # Replace text in the input text file and save the result to the output text file
    markeddown_contnet=replace_text_in_file("coreferences_text.txt", "coreferences_markdown.txt", replacements)


    # 6. We identify sentences in passive voice and conver them to active voice, 
    ## then put the back in the original text.
    ### Debug: We create a csv showing original sentene, new sentence.
    
    # Process sentences
    # Convert Markdown to HTML
    text=get_text_msrkdown(markeddown_contnet)


    input_sentences_list = []
    active_sentence_list = []
    # Split text into sentences
    for text_list in text:
        sentences = nltk.sent_tokenize(text_list)

        # Print each sentence
        for sentence in sentences:
            if is_passive(sentence):
                active_sentence = convert_to_active_voice(sentence)
                active_sentence_list.append(active_sentence)
                input_sentences_list.append(sentence)
                markeddown_contnet = markeddown_contnet.replace(sentence,active_sentence)
    create_debug_csv_file(input_sentences_list, active_sentence_list,"Active Voice Text", "Active_Voice.csv")
    create_debug_txt_file(markeddown_contnet,"active_voice_text.txt")

    # 7. We identify sentences that contain the modal verbs (can, could, may, might, shall, will, would, must) and transform them to non modal sentences, then replace the old sentences.
    ## Debug: We create a csv showing original sentene, new sentence.
    text=get_text_msrkdown(markeddown_contnet)
    # Split text into sentences
    for text_list in text:
        sentences = nltk.sent_tokenize(text_list)

        # Print each sentence
        for sentence in sentences:

            conveted_text = ""
            converted_text=process_sentence(sentence)
            if converted_text==False:
                continue
            while converted_text!=False:
                previous_conveted_text=converted_text
                converted_text=process_sentence(converted_text)
            markeddown_contnet = markeddown_contnet.replace(sentence,previous_conveted_text)
    create_debug_txt_file(markeddown_contnet,"non_modal_verb_text.txt")

    create_debug_csv_file(can_input_list, can_output_list,"Model Verb without can Text", "model_verb_without_can.csv")
    create_debug_csv_file(could_input_list, could_output_list,"Model Verb without could Text", "model_verb_without_could.csv")
    create_debug_csv_file(may_input_list, may_output_list,"Model Verb without may Text", "model_verb_without_may.csv")
    create_debug_csv_file(might_input_list, might_output_list,"Model Verb without might Text", "model_verb_without_might.csv")
    create_debug_csv_file(shall_input_list, shall_output_list,"Model Verb without shall Text", "model_verb_without_shell.csv")
    create_debug_csv_file(will_input_list, will_output_list,"Model Verb without will Text", "model_verb_without_will.csv")
    create_debug_csv_file(would_input_list, would_output_list,"Model Verb without would Text", "model_verb_without_would.csv")
    create_debug_csv_file(must_input_list, must_output_list,"Model Verb without must Text", "model_verb_without_must.csv")
    create_debug_csv_file(should_input_list, should_output_list,"Model Verb without should Text", "model_verb_without_should.csv")

    # 8. We identify sentences that start with "if" and transform them so they don't start with if.
    ## Debug: We create a csv showing original sentene, new sentence.
    text=get_text_msrkdown(markeddown_contnet)
    # Split text into sentences
    input_if_sentences = []
    output_without_if_sentences = []
    for text_list in text:
        sentences = nltk.sent_tokenize(text_list)

        # Print each sentence
        for sentence in sentences:
            if check_if_sentences(sentence)==True:
                converted_text=transform_if_sentences(sentence)
                input_if_sentences.append(sentence)
                output_without_if_sentences.append(converted_text)
                markeddown_contnet = markeddown_contnet.replace(sentence,converted_text)
    create_debug_txt_file(markeddown_contnet,"do_not_start_with_if_text.txt")
    create_debug_csv_file(input_if_sentences, output_without_if_sentences,"Sentence do not start with if", "do_not_start_with_if.csv")

    # 9. We identify sentences that start with "there is", "there are" and transform them so they don't start with "there is", "there are".
    ## Debug: We create a csv showing original sentene, new sentence.

    text=get_text_msrkdown(markeddown_contnet)
    # Split text into sentences
    input_there_is_sentences = []
    output_without_there_is_sentences = []

    input_there_are_sentences = []
    output_without_there_are_sentences  = []
    for text_list in text:
        sentences = nltk.sent_tokenize(text_list)

        # Print each sentence
        for sentence in sentences:
            if check_there_is_sentences(sentence)==True:
                converted_text=transform_there_is_sentences(sentence)
                input_there_is_sentences.append(sentence)
                output_without_there_is_sentences.append(converted_text)
                markeddown_contnet = markeddown_contnet.replace(sentence,converted_text)
            if check_there_are_sentences(sentence)==True:
                converted_text=transform_there_are_sentences(sentence)
                input_there_are_sentences.append(sentence)
                output_without_there_are_sentences.append(converted_text)
                markeddown_contnet = markeddown_contnet.replace(sentence,converted_text)

    create_debug_txt_file(markeddown_contnet,"do_not_start_with_there_is_there_are_text.txt")
    create_debug_csv_file(input_there_is_sentences, output_without_there_is_sentences,"Sentence do not start with there is", "do_not_start_with_there_is.csv")     
    create_debug_csv_file(input_there_are_sentences, output_without_there_are_sentences,"Sentence do not start with there are", "do_not_start_with_there_are.csv")         
    
    # 10. We Generate create a new .md file with based on the old md, but with all the sentences that were modified in the steps.
    create_debug_txt_file(markeddown_contnet,"final_output_text.txt")           
            
if __name__ == "__main__":
    os.environ['OPENAI_API_KEY'] = ""

    client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    )
    
    main()