##Import dependencies
from __future__ import absolute_import
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import csv
import nltk
import os
import re
import sys
import math
import liwc
import textstat
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TextAnalysis:
    def __init__(self, input_file_path, liwc_parser_path, dataset_path, output_file_path):
        self.input_file_path = input_file_path
        self.dataset_path = dataset_path
        self.output_file_path = output_file_path
        self.Final_Vector = []
        self.column_names = []
        self.parse, self.category_names = self.load_liwc_parser(liwc_parser_path)

    @staticmethod
    def load_liwc_parser(liwc_parser_path):
        """
        Load the LIWC parser from the given file path.
        """
        try:
            import liwc
            parse, category_names = liwc.load_token_parser(liwc_parser_path)
            return parse, category_names
        except Exception as e:
            print(f"Exception occurred while loading LIWC parser: {e}")
            return None, None

    @staticmethod
    def tokenize(text):
        """
        Tokenize the given text using a basic tokenizer.
        """
        for match in re.finditer(r'\w+', text, re.UNICODE):
            yield match.group(0)

    @staticmethod
    def get_unique_in_order(original_list):
        seen = set()
        output_list = []
        for item in original_list:
            if item not in seen:
                seen.add(item)
                output_list.append(item)
        return output_list

    # Part 1: Functions to calculate sentiment features
    def polarity_score(self, text, k):
        try:
            #print (text,k)
            pol_score = TextBlob(text)
            val="{:.3f}".format(pol_score.sentiment.polarity)
            #print ('Polarity score:', pol_score.sentiment.polarity)
            self.Final_Vector[k].append(val)
            self.column_names.append("polarity_score")
        except:
            print('Exception occurs in Polarity_Score function')

    def subjectivity_score(self, text, k):
        try:
            subjectivity_score = TextBlob(text)
            val="{:.3f}".format(subjectivity_score.sentiment.subjectivity)
            #print ('Subjectivity score:', subjectivity_score.sentiment.subjectivity)
            self.Final_Vector[k].append(val)
            self.column_names.append("subject_score")
        except:
            print('Exception occurs in Subjectivity_Score function')

    def positive_words_count(self, text, k):
        try:
            pos_word_list=[]
            token = word_tokenize(text)
            #print(token)
            for word in token:
                testimonial = TextBlob(word)
                #print(testimonial)
                if testimonial.sentiment.polarity >= 0.5:
                    pos_word_list.append(word)
                    #print(pos_word_list)

            self.Final_Vector[k].append(len(pos_word_list))
            self.column_names.append("pos_word_cnt")
        except:
            print("Exception occur in Positive_Words_Count function")

    def negative_words_count(self, text, k):
        try:
            neg_word_list=[]
            token = word_tokenize(text)
            #print (token)
            for word in token:
                testimonial = TextBlob(word)
                if testimonial.sentiment.polarity <= -0.5:
                    neg_word_list.append(word)

            self.Final_Vector[k].append(len(neg_word_list))
            self.column_names.append("neg_word_cnt")
        except:
            print("Exception occur in Negative_Words_Count function")

    def Interjection_count(self, row, k):
        try:
            count=0
            text = word_tokenize(row)                
            pos=nltk.pos_tag(text)
            selective_pos = ['UH']
            #for word,tag in pos:
                    #print (tag)
            selective_pos_words = []
            for word,tag in pos:
                    if tag in selective_pos:
                            selective_pos_words.append((word,tag))
                            count+=1
            if (count>2):
                self.Final_Vector[k].append(1)
            else:
                self.Final_Vector[k].append(0)
            self.column_names.append("interject_cnt")
        except:
            print("Exception occur in Interjection_Count function")

    # Part 2: Functions to calculate POS-based features
    def noun_count(self, row, k):
        try:
            count=0
            text = word_tokenize(row)
            pos=nltk.pos_tag(text)
            selective_pos = ['NN']
            selective_pos_words = []
            for word,tag in pos:
                if tag in selective_pos:
                    selective_pos_words.append((word,tag))
                    count+=1
            if (count>2):
                self.Final_Vector[k].append(1)
            else:
                self.Final_Vector[k].append(0)
            self.column_names.append("noun_cnt")
        except:
            print("Exception occur in Noun Count function")

    def verb_count(self, row, k):
        try:
            count=0
            text = word_tokenize(row)
            pos=nltk.pos_tag(text)
            selective_pos = ['VB']
            #for word,tag in pos:
            #print (tag)
            selective_pos_words = []
            for word,tag in pos:
                if tag in selective_pos:
                    selective_pos_words.append((word,tag))
                    count+=1
            if (count>2):
                self.Final_Vector[k].append(1)
            else:
                self.Final_Vector[k].append(0)
            self.column_names.append("verb_cnt")
        except:
            print("Exception occur in Verb Count function")

    def adverb_count(self, row, k):
        try:
            count=0
            text = word_tokenize(row)
            pos=nltk.pos_tag(text)
            selective_pos = ['RB']
            #for word,tag in pos:
            #print (tag)
            selective_pos_words = []
            for word,tag in pos:
                if tag in selective_pos:
                    selective_pos_words.append((word,tag))
                    count+=1
            if (count>2):
                self.Final_Vector[k].append(1)
            else:
                self.Final_Vector[k].append(0)
            self.column_names.append("adverb_cnt")
        except:
            print("Exception occur in Adverb_Count function")

    def adjective_count(self, row, k):
        try:
            count=0
            text = word_tokenize(row)                
            pos=nltk.pos_tag(text)
            selective_pos = ['JJ']
            #for word,tag in pos:
            #print (tag)
            selective_pos_words = []
            for word,tag in pos:
                if tag in selective_pos:
                    selective_pos_words.append((word,tag))
                    count+=1
            if (count>2):
                self.Final_Vector[k].append(1)
            else:
                self.Final_Vector[k].append(0)
            self.column_names.append("adjt_cnt")
        except:
            print("Exception occur in Adjective_count function")

    def affective_valence_score(self, row, k):
        try:
            df = pd.read_csv(r'./Affective/all.csv',delimiter=',',encoding='latin-1')        
            res=0
            token = word_tokenize(row)                
            for word in token:                
                    df1=(df['Valence Mean'].loc[df['Description'] == word])                       
                    for line in list(df1):
                            res+=line                                
            #res=res/len(token)
            self.Final_Vector[k].append(res)
            self.column_names.append("aff_valence_score")
        except:
            print('Exception occurs in Affective_Valence_Score function')

    def affective_arousal_score(self, row, k):
        try:
            df = pd.read_csv(r'./Affective/all.csv',delimiter=',',encoding='latin-1')        
            res=0
            token = word_tokenize(row)                
            for word in token:                
                    df1=(df['Arousal Mean'].loc[df['Description'] == word])                       
                    for line in list(df1):
                            res+=line                                
            #res=res/len(token)
            self.Final_Vector[k].append(res)
            self.column_names.append("aff_arousal_score")
        except:
            print('Exception occurs in Affective_Arousal_Score function')
            
    def affective_dominance_score(self, row, k):
        try:
            df = pd.read_csv(r'./Affective/all.csv',delimiter=',',encoding='latin-1')        
            res=0
            token = word_tokenize(row)                
            for word in token:                
                    df1=(df['Dominance Mean'].loc[df['Description'] == word])                       
                    for line in list(df1):
                            res+=line                                
            #res=res/len(token)
            self.Final_Vector[k].append(res)
            self.column_names.append("aff_dominance_score")
        except:
            print('Exception occurs in Affective_Dominance_Score function')

    #This function is used to calcualte the presence of terms from mental health lexicon,and also unigrams and n-grams count.
    def lexicon_features(self, row, k):
        df = pd.read_csv(r'./Lexicon/Mental_health_lexicon.csv',delimiter=',',encoding='latin-1')
        
        # Apply the cleaning function to the 'keywords' column
        df['keywords'] = df['keywords'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x).lower()).strip())
        df.to_csv("./Lexicon/lexicon_updated.csv")
        
        unigram_count = 0
        ngram_count = 0
        
        token = word_tokenize(row)                
        matched_keywords = []

        # Get the list of cleaned keywords from the DataFrame
        keyword_list = df['keywords'].tolist()

        # Join the tokenized text into a string for easier n-gram matching
        text_str = ' '.join(token)

        # Loop through the keywords and check for matches
        for keyword in keyword_list:
            # Count occurrences of the keyword in the text
            count = text_str.count(keyword)
            if count > 0:
                matched_keywords.extend([keyword] * count)

        # Count occurrences of each matched keyword
        keyword_counts = Counter(matched_keywords)
        #print(keyword_counts)

        # Separate counts for unigrams and bigrams
        unigram_count = sum(count for keyword, count in keyword_counts.items() if len(keyword.split()) == 1)
        bigram_count = sum(count for keyword, count in keyword_counts.items() if len(keyword.split()) >= 2)
        self.Final_Vector[k].append(int(unigram_count>0 or ngram_count>0))
        self.Final_Vector[k].append(unigram_count)
        self.Final_Vector[k].append(ngram_count)
        self.column_names.append("lexicon_present")
        self.column_names.append("lexicon_unigram")
        self.column_names.append("lexicon_ngram")
        

    def pos_stylistics(self, row, k):
        try:
                count_noun=0
                count_pronoun=0
                count_verb=0
                count_adjective=0
                count_adverb=0
                count_determiner=0
                count_conjuction=0
                count_cardinal=0
                count_existential=0
                count_foreign =0
                count_listitem =0
                count_modal =0
                count_possessive =0
                count_particle =0
                count_symbol =0
                count_to =0
                count_interjection =0
                                
                text = word_tokenize(row)
                #print (nltk.pos_tag(text))
                pos=nltk.pos_tag(text)                
                pos_noun = ['NN','NNS','NNP','NNPS']
                pos_pronoun = ['PRP','PRP$','WP','Wp$']
                pos_verb = ['VB','VBD','VBG','VBN','VBP','VBZ']
                pos_adjective = ['JJ','JJR','JJS']
                pos_adverb = ['RB','RBR','RBS','WRB']
                pos_determiner = ['DT','PDT','WDT']
                pos_conjuction = ['CC','IN']
                pos_cardinal = ['CD']
                POS_existential = ['EX']
                POS_foreign = ['FW']
                POS_listitem = ['LS']
                POS_modal = ['MD']
                POS_possessive = ['POS']
                POS_particle = ['RP']
                POS_symbol = ['SYM']
                POS_to = ['TO']
                POS_interjection = ['UH']
                
                for word,tag in pos:
                        if tag in pos_noun:
                                count_noun+=1
                        elif tag in pos_pronoun:
                                count_pronoun+=1
                        elif tag in pos_verb:
                                count_verb+=1
                        elif tag in pos_adjective:
                                count_adjective+=1
                        elif tag in pos_adverb:
                                count_adverb+=1
                        elif tag in pos_determiner:
                                count_determiner+=1
                        elif tag in pos_conjuction:
                                count_conjuction+=1
                        elif tag in pos_cardinal:
                                count_cardinal+=1
                        elif tag in POS_existential:
                                count_existential+=1
                        elif tag in POS_foreign:
                                count_foreign+=1
                        elif tag in POS_listitem:
                                count_listitem+=1
                        elif tag in POS_modal:
                                count_modal+=1
                        elif tag in POS_possessive:
                                count_possessive+=1                        
                        elif tag in POS_particle:
                                count_particle+=1
                        elif tag in POS_symbol:
                                count_symbol+=1
                        elif tag in POS_to:
                                count_to+=1
                        elif tag in POS_interjection:
                                count_interjection+=1
                
                self.Final_Vector[k].append(count_noun)
                self.Final_Vector[k].append(count_pronoun)
                self.Final_Vector[k].append(count_verb)
                self.Final_Vector[k].append(count_adjective)                
                self.Final_Vector[k].append(count_adverb)
                self.Final_Vector[k].append(count_determiner)
                self.Final_Vector[k].append(count_conjuction)
                self.Final_Vector[k].append(count_cardinal)
                self.Final_Vector[k].append(count_existential)
                self.Final_Vector[k].append(count_foreign)
                self.Final_Vector[k].append(count_listitem)
                self.Final_Vector[k].append(count_modal)
                self.Final_Vector[k].append(count_possessive)
                self.Final_Vector[k].append(count_particle)
                self.Final_Vector[k].append(count_symbol)
                self.Final_Vector[k].append(count_to)
                self.Final_Vector[k].append(count_interjection)
                self.column_names.append("pos_count_noun")
                self.column_names.append("pos_count_pronoun")
                self.column_names.append("pos_count_verb")
                self.column_names.append("pos_count_adjective")                
                self.column_names.append("pos_count_adverb")
                self.column_names.append("pos_count_determiner")
                self.column_names.append("pos_count_conjuction")
                self.column_names.append("pos_count_cardinal")
                self.column_names.append("pos_count_existential")
                self.column_names.append("pos_count_foreign")
                self.column_names.append("pos_count_listitem")
                self.column_names.append("pos_count_modal")
                self.column_names.append("pos_count_possessive")
                self.column_names.append("pos_count_particle")
                self.column_names.append("pos_count_symbol")
                self.column_names.append("pos_count_to")
                self.column_names.append("pos_count_interjection")
        except:
                print("Exception occur in POS_Stylistics function")

    def readability_score_features(self, row, k):  
        try:
            self.Final_Vector[k].append(textstat.flesch_reading_ease(row))
            self.Final_Vector[k].append(textstat.gunning_fog(row))
            self.Final_Vector[k].append(textstat.automated_readability_index(row))
            self.Final_Vector[k].append(textstat.coleman_liau_index(row))
            self.Final_Vector[k].append(textstat.syllable_count(row))
            self.column_names.append("readability_flesch_reading")
            self.column_names.append("readability_gunning_fog")
            self.column_names.append("readability_automated_readability_index")
            self.column_names.append("readability_coleman_liau_index")
            self.column_names.append("readability_syllable_count")
        except:
            print('Exception occurs in readabilty_score_features function')

    def structural_features(self, row, k):
        try:
            cnt_exclm=0
            cnt_dots=0
            cnt_ques_marks=0
            cnt_upper=0
            numbers = sum(c.isdigit() for c in row)
            #Final_Vector[k].append(numbers)
            cap_letters = sum(c.isupper() for c in row)
            #Final_Vector[k].append(cap_letters)
            
            pre_text = re.sub(r'[0-9]+', '', row) # Remove digits
            pre_text= pre_text.replace("!", "") # Remove exclamation marks
            pre_text= pre_text.replace(".", "") # Remove dots
            pre_text= pre_text.replace("?", "") # Remove question marks        
            token = word_tokenize(pre_text) 
            
            #self.Final_Vector[k].append(len(token))        
            #self.Final_Vector[k].append(math.log(len(token),5))
            #self.column_names.append("struct_token_len")
            #print (row)
            for i in range (0, len (row)):
                if row[i] in ('!'):
                    cnt_exclm = cnt_exclm + 1                            
                elif row[i] in ('.'):
                    cnt_dots = cnt_dots + 1
                elif row[i] in ('?'):
                    cnt_ques_marks = cnt_ques_marks + 1                

            #print (cnt_exclm,cnt_dots,cnt_ques_marks)      
            self.Final_Vector[k].append(cnt_exclm)
            self.Final_Vector[k].append(cnt_dots)
            self.Final_Vector[k].append(cnt_ques_marks)
            self.column_names.append("struct_cnt_exclm")
            self.column_names.append("struct_cnt_dots")
            self.column_names.append("struct_cnt_ques_marks")
        except:
            print("Exception occur in Structural_Features function")

    def psycholinguistic_liwc(self, row, k):
        #try:        
            #print (row)
            cnt_ling=0
            cnt_psyc=0
            cnt_pers=0
            cnt_spk=0
            linguistic=['funct','pronoun','ppron','i','we','you','shehe','they','ipron','article','verb','auxverb','past','present','future','adverb','preps','conj','negate','quant','number','swear']
            psychological=['social','family','friend','humans','affect','posemo','negemo','anx','anger','sad','cogmech','insight','cause','discrep','tentat','certain','inhib','incl','excl','percept','see','hear','feel','bio','body','health','sexual','ingest','relativ','motion','space','time']
            personal = ['work','achieve','leisure','home','money','relig','death']
            spoken  = ['assent','nonflu','filler']
            
            row_tokens = self.tokenize(row)
            row_counts = Counter(category for token in row_tokens for category in self.parse(token))
            #print(dict(row_counts))
            
            for l in linguistic:
                if l in dict(row_counts):
                    ling=dict(row_counts)[l]
                    cnt_ling=cnt_ling+ling
            #print(cnt_ling)                
            self.Final_Vector[k].append(cnt_ling)
            
            for p in psychological:
                if p in dict(row_counts):
                    psyc=dict(row_counts)[p]
                    cnt_psyc=cnt_psyc+psyc
            #print (cnt_psyc)
            self.Final_Vector[k].append(cnt_psyc)
            
            for per in personal:
                if per in dict(row_counts):
                    pers=dict(row_counts)[per]
                    cnt_pers=cnt_pers+pers
            #print (cnt_pers)
            self.Final_Vector[k].append(cnt_pers)

            for s in spoken:
                if s in dict(row_counts):
                    spk=dict(row_counts)[s]
                    cnt_spk=cnt_spk+spk
            #print (cnt_spk)
            self.Final_Vector[k].append(cnt_spk)

            self.column_names.append("psych_LIWC_linguistic")
            self.column_names.append("psych_LIWC_psychological")
            self.column_names.append("psych_LIWC_personal")
            self.column_names.append("psych_LIWC_spoken")
        #except:
            #print("Exception occur in Psycholinguistic_LIWC function")

    def process_dataset(self):
        """
        Process the dataset by applying all feature calculation functions.
        """
        try:
            # Load dataset
            with open(self.dataset_path + '.csv', 'r') as file:
                reader = csv.reader(file, quoting=csv.QUOTE_ALL)
                headers = next(reader)
                tag_list = list(reader)

            # Initialize final vector
            for _ in tag_list:
                self.Final_Vector.append([])

            # Apply feature extraction functions
            for k, sublist in enumerate(tag_list):
                text = ' '.join(map(str, sublist))
                self.polarity_score(text, k)
                self.subjectivity_score(text, k)
                self.positive_words_count(text, k)
                self.negative_words_count(text, k)
                self.noun_count(text, k)
                self.verb_count(text, k)
                self.adverb_count(text, k)
                self.adjective_count(text, k)
                self.readability_score_features(text, k)
                self.affective_valence_score(text, k)
                self.affective_arousal_score(text, k)
                self.affective_dominance_score(text, k)
                #self.pos_stylistics(text, k)
                self.psycholinguistic_liwc(text, k)
                self.structural_features(text, k)
                self.lexicon_features(text, k)
                self.Final_Vector[k].append(sublist[4])  # Append label column from input csv file which is 5th column in csv whereas in this term [0 1 2 3 4] index of col is 4

                if k % 100 == 0:
                    print(f"Processed {k} records.")
        except Exception as e:
            print(f"Exception occurred while processing dataset: {e}")

    def save_feature_vectors(self):
        """
        Save the processed feature vectors to a CSV file.
        """
        try:
            with open(self.output_file_path + '_fvect27_all_v1.csv', 'w') as file:
                writer = csv.writer(file)
                column_names_new = self.get_unique_in_order(self.column_names)
                column_names_new.append('label')
                writer.writerow(column_names_new)
                writer.writerows(self.Final_Vector)
            print("Feature vectors saved successfully.")
        except Exception as e:
            print(f"Exception occurred while saving feature vectors: {e}")

    @staticmethod
    def normalize_columns(file_path, columns_to_normalize):
        """
        Normalize specified columns in the CSV file.
        """
        try:
            df = pd.read_csv(file_path)
            for column in columns_to_normalize:
                df[column] = round((df[column] - df[column].min()) / (df[column].max() - df[column].min()), 2)
            temp_path = file_path.split(".csv")[0]
            temp_path = temp_path + "_normalized.csv"
            df.to_csv(temp_path, index=False)
            print("Normalization complete and file saved.")
        except Exception as e:
            print(f"Exception occurred during normalization: {e}")

    @staticmethod
    def scale_and_save_dataset(file_path, output_file, scaler_type="standard"):
        """
        Scale the dataset using the specified scaler (StandardScaler or MinMaxScaler) and save it.
        """
        try:
            df = pd.read_csv(file_path)
            df = df.reset_index(drop=True)
            #df = df.drop(df.columns[0], axis=1)

            # Choose scaler
            if scaler_type == "standard":
                scaler = StandardScaler()
            elif scaler_type == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError("Invalid scaler type. Choose 'standard' or 'minmax'.")

            # Scale the data
            scaled_columns = scaler.fit_transform(df)

            # Create a DataFrame from the transformed columns
            scaled_df = pd.DataFrame(
                scaled_columns,
                columns=df.columns,
                index=df.index
                )

            # Save scaled DataFrame
            scaled_df.to_csv(output_file, index=False)
            print(f"Scaled dataset ({scaler_type}) saved successfully.")
        except Exception as e:
            print(f"Exception occurred during dataset scaling: {e}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process text data and generate feature vectors.")
    parser.add_argument(
        "--input_dataset_path", 
        type=str, 
        required=True, 
        help="Path to the input dataset (without extension). Example: './dataset/Kaggle_20_val'"
        )

##    parser.add_argument(
##        "--liwc_parser_path",
##        type=str,
##        required=True,
##        help="Path to the LIWC dictionary file. Example: './LIWC2007_English100131.dic'"
##    )
    

    # Parse command-line arguments
    #input_dataset_path = "./dataset/Kaggle_20_val.csv" #take input from argparser
    args = parser.parse_args()
    input_dataset_path = args.input_dataset_path
    #print(input_dataset_path)
    
    # Define file paths
    #liwc_parser_path = args.liwc_parser_path
    liwc_parser_path = "./LIWC/LIWC2007_English100131.dic"

    dataset_path = input_dataset_path.split(".csv")[0]
    output_file_path = dataset_path + "_output"

    # Columns to normalize (example column names; replace with actual column names from your dataset)
    columns_to_normalize = [
        "readability_flesch_reading",
        "readability_gunning_fog",
        "readability_automated_readability_index",
        "readability_coleman_liau_index",
        "readability_syllable_count",
        "aff_valence_score",
        "aff_arousal_score",
        "aff_dominance_score",
        "psych_LIWC_linguistic",
        "psych_LIWC_psychological",
        "psych_LIWC_personal"
        ]

    # Initialize the TextAnalysis class
    text_analysis = TextAnalysis(
        input_file_path=input_dataset_path,
        liwc_parser_path=liwc_parser_path,
        dataset_path=dataset_path,
        output_file_path=output_file_path
    )

    # Process the dataset to create feature vectors
    print("Processing dataset...")
    text_analysis.process_dataset()

    # Save the feature vectors
    print("Saving feature vectors...")
    text_analysis.save_feature_vectors()

    # Normalize the specified columns in the feature vector file
    output_final_path = output_file_path + "_fvect27_all_v1.csv"
    print("Normalizing columns...")
    text_analysis.normalize_columns(output_final_path, columns_to_normalize)

    # Scale the normalized dataset using StandardScaler and MinMaxScaler
    standard_scaled_output = output_file_path + "_standardscaled_v1.csv"
    minmax_scaled_output = output_file_path + "_minmaxscaled_v1.csv"

    print("Scaling dataset with StandardScaler...")
    text_analysis.scale_and_save_dataset(output_final_path, standard_scaled_output, scaler_type="standard")

    print("Scaling dataset with MinMaxScaler...")
    text_analysis.scale_and_save_dataset(output_final_path, minmax_scaled_output, scaler_type="minmax")

    print("Processing complete!")

