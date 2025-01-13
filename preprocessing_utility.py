#libraries
import re
import emoji
from html import unescape
from nltk.tokenize import sent_tokenize, word_tokenize
import unicodedata
import pandas as pd


CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

def clean_text(text):
    # Remove emojis
    text = emoji.get_emoji_regexp().sub(r'', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove HTTP links (with and without https://)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove usernames
    text = re.sub(r'@(\w+)', '', text)
    
    # Remove extra whitespaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocessing_pipeline(data):
    context_clean = list()
    data = re.sub(r"\\n", r' ', str(data))
    data = ' '.join(data.split())
    data = sent_tokenize(data)
    for sent in data:
        sent = str(sent)
        sent = remove_http_links(sent)
        sent = whitespace(sent)
        sent = remove_accented(sent)
        sent = expand_contractions(sent, CONTRACTION_MAP)
        sent = remove_numbers_except_dollar(sent)
        sent = text_clean(sent)
        sent = remove_dates(sent)
        #sent = remove_num_punkt(sent)
        context_clean.append(sent.lower())
#    return context_clean
    return " ".join(context_clean)

def remove_numbers_except_dollar(string):
    # Define a regular expression pattern to match numbers not preceded by $
    pattern = r'(?<!\$)\b\d+(\.\d+)?\b'
    # Use re.sub() to replace matched numbers with an empty string
    result = re.sub(pattern, '', string)
    return result

def whitespace(sentence):
    sentence = ' '.join(sentence.split())
    sentence = re.sub(r"-", r" ", sentence)
    sentence = re.sub(r"/", r" ", sentence)
    sentence = re.sub(r"_", r" ", sentence)
    sentence = re.sub(r",", r" ", sentence)
    sentence = re.sub(r" NA ", r"", sentence)
    sentence = re.sub(r"%", r"percent", sentence)
    #sentence = re.sub(r".", r" ", sentence)
    #text_lower = sentence.lower()
    text_lower = sentence
    return text_lower

def remove_http_links(text):
    # Regular expression to match HTTP links
    http_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    # Remove HTTP links from the text
    cleaned_text = re.sub(http_pattern, '', text)

    return cleaned_text

def tokenize(sentence):
    text_token = word_tokenize(sentence)
    return text_token


def remove_accented(sentence):
    text_ascii = unicodedata.normalize('NFKD', sentence).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text_ascii


def expand_contractions(sentence, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, sentence)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_num_punkt(sentence):
    text_alpha = tokenize(sentence)
    text_alpha = [word for word in text_alpha if (word.isalnum() or "$" in word)]
    text_alpha = ' '.join(text_alpha)
    return text_alpha

def remove_dates(string):
    # Define regular expression patterns for different date formats
    date_patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',  # Matches YYYY-MM-DD
        r'\b\d{4}/\d{2}/\d{2}\b',   # Matches YYYY/MM/DD
        r'\b\d{1,2}(st|nd|rd|th)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b'  # Matches dates like "29th August"
    ]
    
    # Combine all patterns into a single pattern
    combined_pattern = '|'.join(date_patterns)
    
    # Use re.sub() to replace matched date patterns with an empty string
    result = re.sub(combined_pattern, '', string)
    
    return result


def text_clean(sentence):
    #sentence = 'cls ' + sentence + ' eos'
    #sentence = sentence.lower()
    #date_time= re.findall(r'\b[\d]{1,4}[/:][\d]{1,4}[/:][\d]{1,4}\b',sentence)
    #for i in date_time:
    #    sentence = re.sub(i,"",sentence)
    #special_char = '''/\\#,"?()!:@<>$~%^&*;%[].{}-'â€™_`=â€'''
    special_char = '''|/\\#,"?()!:@<>~^&*;.[]{}'â€™_`=â€'''
    
    token_clean = re.sub(r"\\n", r' ', sentence)
    token_clean = re.sub(r" NA ", r"", token_clean)
    for char in special_char:
        token_clean = token_clean.replace(char, ' ')
        
    token_clean = ' '.join(token_clean.split())
    token_clean = re.sub(r'(\d+)\s+(?=\d)', r'\1', token_clean)
    #token_clean = token_clean.lower()

    return token_clean

#path = "/path/to/excel_file.xlsx"
#file_name = path.split(".xlsx")[0]
#df = pd.read_excel(path)

path = "/path/to/csv_file.csv"
file_name = path.split(".csv")[0]
df = pd.read_csv(path)

df = df.dropna()

df['cleaned_text'] = df['text'].apply(lambda x: clean_text(str(x)))
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: preprocessing_pipeline(x))


#label conversion if required {'TRUE': 1, 'FALSE': 0, 'neutral': -1} (remove neutral samples from dataset)
df["lable_converted"] = df["label"].replace({'TRUE': 1, 'FALSE': 0, 'neutral': -1})

df.to_csv(file_name+"_preprocessed.csv")
