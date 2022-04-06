# #!/usr/bin/env python

# # -*- coding: utf-8 -*-


# Most code is based on https://melaniewalsh.github.io/Intro-Cultural-Analytics/features/Text-Analysis/Named-Entity-Recognition.html


import math
import re
import spacy
import csv
from spacy import displacy
from collections import Counter
import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

pd.options.display.max_rows = 600
pd.options.display.max_colwidth = 400
pd.set_option("display.max_rows", None, "display.max_columns", None)

import nl_core_news_lg
nlp = nl_core_news_lg.load()
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

from spacy.tokens import Doc


from spacy.tokens import Span
from spacy.util import filter_spans



# READ CSV FILE 


csvpath = '/Users/roelsmeets/desktop/complot/Telegram_data'
csvpath2 = '/Users/roelsmeets/desktop/complot/Telegram_data_done'
csvpath3 = '/Users/roelsmeets/desktop/complot/Telegram_data_to_do'
csvfiles = {}

filepath = '/Users/roelsmeets/desktop/complot/boereninopstand.csv'
#text = open(filepath, encoding='utf-8').read()



def remove_images(post):
    '''  Verwijder: 
                alles tussen < > (zijn afbeeldingen of andere media)
    

    '''
 
    clean_post = re.sub(r'<U.+?>', '', post) # Remove al images, i.e. everything between <U ... >
      

    return (clean_post)


def remove_forwards(clean_post):
    '''  Verwijder: 
                    Alles tussen list(...) , dat zijn links naar filmpjes
            
            Discussiepunt: moet dit weg?
            Gaat niet helemaal goed nog, regex moet aangepast worden.

    '''

    cleaner_post = re.sub(r'list\(.+?\)', '', clean_post) # Remove all forwards i.e. everything between list(.... )... >
               

    return (cleaner_post)


def remove_links(cleaner_post):
    '''  Verwijder: 
                    alles tussen list(type...) , dat zijn links naar filmpjes

        Gaat niet helemaal goed nog, regex moet aangepast worden.
        

     '''
    

    cleanest_post = re.sub(r'list\(type.+?\)', '', cleaner_post) # Remove all links, i.e. everything between list(.... )... >
          

    return (cleanest_post)





def delete_breaks(cleanest_post):
    '''  Verwijder: 
                regelafbrekingen \n 
    
        Gaat niet helemaal goed nog (vooral als het vastzit aan een woord), misschien beter met regex. Nu gewoon via Search-Replace verwijderd in Sublime.

     '''
    cleanests_post = cleanest_post.strip()
    cleanests_post = cleanest_post.replace("\n"," ") 
    cleanests_post = cleanest_post.replace("\n\n"," ") 
    cleanests_post = cleanest_post.replace("\\n"," ") 
    cleanests_post = cleanest_post.replace("\\n\\n"," ") 


    return (cleanests_post)


def listToString(s): 
        
    # initialize an empty string
    str1 = " " 
        
    # return string  
    return (str1.join(s))


def match_patterns(cleanests_post):

    """
    Hardcode co-references of a range of characters.

    For some very frequent characters, the OP parameter is used to capture both first and last name individually and combined. 
    This is based on qualitative assessment of each character. 

    """


    mark_rutte = [
    [{"LOWER": "mark", 'OP': '?'}, {"LOWER": "rutte", 'OP': '?'}],

    [{"LOWER": "markie"}],

    [{"LOWER": 'murk ratte'}]

    ]
    matcher.add("Mark Rutte", mark_rutte, on_match=add_person_ent)


    hugo_dejonge = [
    [{"LOWER": "hugo", 'OP': '?'}, {"LOWER": "de jonge", 'OP': '?'}]

    ]
    matcher.add("Hugo de Jonge", hugo_dejonge, on_match=add_person_ent)


    sigrid_kaag = [
    [{"LOWER": "sigrid", 'OP': '?'}, {"LOWER": "kaag", 'OP': '?'}]

    ]
    matcher.add("Sigrid Kaag", sigrid_kaag, on_match=add_person_ent)


    geert_wilders =  [
    [{"LOWER": "geert", 'OP': '?'}, {"LOWER": "wilders", 'OP': '?'}],

    [{"LOWER": "geertje"}]

    ]
    matcher.add("Geert Wilders", geert_wilders, on_match=add_person_ent)


    adolf_hitler = [
    [{"LOWER": "adolf", 'OP': '?'}, {"LOWER": "hitler", 'OP': '?'}]

    ]
    matcher.add("Adolf Hitler", adolf_hitler, on_match=add_person_ent)

   
    donald_trump = [
    [{"LOWER": "donald", 'OP': '?'}, {"LOWER": "trump", 'OP': '?'}]

    ]
    matcher.add("Donald Trump", donald_trump, on_match=add_person_ent)


    thierry_baudet = [
    [{"LOWER": "thierry", 'OP': '?'}, {"LOWER": "baudet", 'OP': '?'}]

    ]
    matcher.add("Thierry Baudet", thierry_baudet, on_match=add_person_ent)


    klaus_schwab =  [
    [{"LOWER": "klaus", 'OP': '?'}, {"LOWER": "schwab", 'OP': '?'}],

    [{"LOWER": "klaus"}],  [{"LOWER": "swaab"}],

    [{"LOWER": "klaus"}],  [{"LOWER": "schwaab"}],

    ]
    matcher.add("Klaus Schwab", klaus_schwab, on_match=add_person_ent)


    bill_gates =  [
    [{"LOWER": "bill", 'OP': '?'}, {"LOWER": "gates", 'OP': '?'}],

    [{"LOWER": "bil"}, {"LOWER": "gates"}]

    ]
    matcher.add("Bill Gates", bill_gates, on_match=add_person_ent)


    george_soros =  [
    [{"LOWER": "george"}, {"LOWER": "soros"}],

    [{"LOWER": "soros"}]

    ]
    matcher.add("George Soros", george_soros, on_match=add_person_ent)


    jan_huzen =  [
    [{"LOWER": "jan"}, {"LOWER": "huzen"}],

    [{"LOWER": "jan"}, {"LOWER": "hunze"}]

    ]
    matcher.add("Jan Huzen", jan_huzen, on_match=add_person_ent)


    john_demol = [
    [{"LOWER": "john"}, {"LOWER": "de mol"}]

    ]
    matcher.add("John de Mol", john_demol, on_match=add_person_ent)


    hubert_bruls = [    
    [{"LOWER": "hubert", 'OP': '?'}, {"LOWER": "bruls", 'OP': '?'}],

    ]
    matcher.add("Hubert Bruls", hubert_bruls, on_match=add_person_ent)


    gideon_vanmeijeren = [    
    [{"LOWER": "gideon", 'OP': '?'}, {"LOWER": "van meijeren", 'OP': '?'}],

    ]
    matcher.add("Gideon van Meijeren", gideon_vanmeijeren, on_match=add_person_ent)


    willem_engel = [    
    [{"LOWER": "willem engel"}],

    [{"TEXT": "Engel"}]

    ]
    matcher.add("Willem Engel", willem_engel, on_match=add_person_ent)



    vladimir_putin = [    
    [{"LOWER": "putin"}],

    [{"LOWER": "poetin"}],

    [{"LOWER": "vladimir poetin"}],

    [{"LOWER": "vladimir putin"}],

    ]
    matcher.add("Vladimir Putin", vladimir_putin, on_match=add_person_ent)

  
    matches = matcher(cleanests_post)

    matches.sort(key = lambda x:x[1])

    new_ents = []

    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        new_ents.append(string_id)
        span = cleanests_post[start:end]  # The matched span
        # print('matches', match_id, string_id, start, end, span.text)
        # print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    
    return (new_ents) # Add new_ents to Doc object (cleanests_post) later


    #print([token.text for token in doc])


def add_person_ent(matcher, cleanests_post, i, matches):
        
    # Get the current match and create tuple of entity label, start and end.
    # Append entity to the doc's entity. (Don't overwrite doc.ents!)

    match_id, start, end = matches[i]
    entity = Span(cleanests_post, start, end, label="PERSON")


    filtered = filter_spans(cleanests_post.ents) # When spans overlap, the (first) longest span is preferred over shorter spans.

    filtered += (entity,)

    return (filtered)

    # for ent in filtered:
    #     print (ent.text, ent.label_)
    #     print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')


def overwrite_doc(cleanests_post):
    # Generate a new list of tokens here
    # This should be added to the new Doc object overwritten_post
    # Strategy: access 'words' in cleanests_post (= Doc object) and use that for words=new_words
    # https://stackoverflow.com/questions/57187116/how-to-modify-spacy-tokens-doc-doc-tokens-with-pipeline-components-in-spacy/57206316#57206316 

    new_words = cleanests_post()

    overwritten_post = Doc(cleanests_post.vocab, words=new_words)

    return overwritten_post




for filename in os.listdir('/Users/roelsmeets/desktop/telegram_narratives/complot/Telegram_data_to_do'):
    if filename.endswith('.csv'):
        print ('computing NLP for:', filename)
        with open(os.path.join('/Users/roelsmeets/desktop/telegram_narratives/complot/Telegram_data_to_do', filename), encoding='latin-1') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')

            next(reader, None) # Skip first row (= header) of the csv file

            dict_from_csv = {rows[0]:rows[2] for rows in reader} # creates a dictionary with 'date' as keys and 'text' as values
            #print ('dict:', dict_from_csv)

            values = dict_from_csv.values()
            values_list = list(values)
      
            people = [] 

            narrative_string = '' # store all posts as one big string


            for post in values_list: # iterate over each post
               

                # Do some preprocessing here  

                clean_post = remove_images(post)

                cleaner_post = remove_forwards(clean_post)

                cleanest_post = remove_links(cleaner_post)

                cleanests_post = delete_breaks(cleanest_post)

                narrative_string = narrative_string + '. ' + cleanests_post


            #print ('posts as string:', narrative_string) 

            print ('length of', filename, 'in characters:', len(narrative_string))

            nlp.max_length = len(narrative_string) + 100

            cleanests_post = nlp(narrative_string, disable = ['parser'])

            new_ents = match_patterns(cleanests_post) 

            # if cleanests_post.ents:
            #     show_results = displacy.render(cleanests_post, style='ent')
            # # print ('result:', show_results)
            # # print ('************************')
            # print ('result ents:', cleanests_post.ents)
            # print ('************************')

            # for named_entity in cleanests_post.ents:
            #     print('named entity:', named_entity, 'label:', named_entity.label_)
            #     print ('************************')


            # for named_entity in cleanests_post.ents:
            #   if named_entity.label_ == "PERSON":
            #       print(named_entity, '= PERSON')


            # GET PEOPLE   

            #print ('new_ents:', new_ents)     

            for named_entity in cleanests_post.ents:
                if named_entity.label_ == "PERSON":
                    #print ('NE PERSON:', named_entity)
                    people.append(named_entity.text)

            for ent in new_ents:
                people.append(ent)

            #print ('people:', people)

            people_tally = Counter(people)

            df = pd.DataFrame(people_tally.most_common(), columns=['character', 'count'])
            #print ('people:', df)

            df.to_csv(filename + '_ranking.csv', index=False)

            # WRITE TO CSV

            # with open ('Telegram_mastersheet.csv', 'a', newline='') as f:
            #     csvwriter = csv.writer(f)

            #     csvwriter.writerow([]) # Key of dict_from_csv (= date), value of dict_from_csv (= post), author (= new dict_from_csv with author as key and post as value), identified PERS ents (+ new ents)



            # # ESTIMATE GENDER OF PERSON

            # filepath1 = '/Users/roelsmeets/Desktop/actual_fictions/actual-fictions/male_names_dutch.csv'
            # filepath2 = '/Users/roelsmeets/Desktop/actual_fictions/actual-fictions/female_names_dutch.csv'

            # column_names1 = ['Name_male', 'Occurrences']
            # male_names_csv = pd.read_csv(filepath1, sep=";", names=column_names1)

            # column_names2 = ['Name_female', 'Occurrences']
            # female_names_csv = pd.read_csv(filepath2, sep=";", names=column_names2)

            # male_names = male_names_csv.Name_male.to_list()
            # female_names = female_names_csv.Name_female.to_list()

            # # print (male_names)
            # # print (female_names)

            # people_list = df.character.to_list()




            # # with open ('characternames.csv', 'a', newline='') as f:
            # #     csvwriter = csv.writer(f)
            # #     """

            # #     Columns: book_id, character_id, ner_rank, character_name, estimated_gender

            # #     In for-if loop below, define the values of the rows that have to be created by csvwriter.writerow 

            # #     """



            # #     for person in people_list[:30]:
            # #         if person in male_names and person in female_names:
            # #             print ('RANK:', people_list.index(person), 'CHARACTER:', person, 'NAME', '= gender neutral name')
            # #             # csvwriter.writerow(book_id, character_id, ner_rank, character_name, estimated_gender )
            # #         elif person in male_names and person not in female_names:
            # #             print ('RANK:', people_list.index(person), 'CHARACTER:', person, 'NAME',  '= probably male')
            # #         elif person in female_names and person not in male_names:
            # #             print ('RANK:', people_list.index(person), 'CHARACTER:', person, 'NAME', '= probably female')
            # #         else:
            # #             print ('RANK:', people_list.index(person), 'CHARACTER:', person, 'NAME', '= strange name OR entity is not a person')


                



            # # GET PLACES

            # places = []

            # for named_entity in document.ents:
            #   if named_entity.label_ == "GPE" or named_entity.label_ == "LOC":
            #       places.append(named_entity.text)

            # places_tally = Counter(places)

            # df = pd.DataFrame(places_tally.most_common(), columns=['place', 'count'])
            # print ('places:', df)



            # # GET STREETS, PARKS, ET CETERA

            # # streets = []

            # # for named_entity in document.ents:
            # #   if named_entity.label_ == "FAC":
            # #       streets.append(named_entity.text)

            # # streets_tally = Counter(streets)

            # # df = pd.DataFrame(streets_tally.most_common(), columns = ['street', 'count'])
            # # print ('streets, parks, etc:', df)



            # # # GET WORKS OF ART 

            # # works_of_art = []

            # # for named_entity in document.ents:
            # #   if named_entity.label_ == "WORK_OF_ART":
            # #       works_of_art.append(named_entity.text)

            # # art_tally = Counter(works_of_art)

            # # df = pd.DataFrame(art_tally.most_common(), columns = ['work_of_art', 'count'])
            # # print ('works of art:', df)




            # # FUNCTION FOR GETTING NER IN CONTEXT OF TEXT


            # def get_ner_in_context(keyword, document, desired_ner_labels= 'PERSON'):
            #     if desired_ner_labels != False:
            #         desired_ner_labels = desired_ner_labels
            #     else:
            #         desired_ner_labels = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']  

            #     #Iterate through all the sentences in the document and pull out the text of each sentence
            #     for sentence in document.sents:
            #         #process each sentence
            #         sentence_doc = nlp(sentence.text)

            #         for named_entity in sentence_doc.ents:
            #             #Check to see if the keyword is in the sentence (and ignore capitalization by making both lowercase)
            #             if keyword.lower() in named_entity.text.lower()  and named_entity.label_ in desired_ner_labels:
            #                 #Use the regex library to replace linebreaks and to make the keyword bolded, again ignoring capitalization
            #                 #sentence_text = sentence.text

            #                 sentence_text = re.sub('\n', ' ', sentence.text)
            #                 sentence_text = re.sub(f"{named_entity.text}", f"**{named_entity.text}**", sentence_text, flags=re.IGNORECASE)

            #                 print('---')
            #                 print(f"**{named_entity.label_}**")
            #                 print(sentence_text)
                   

            # #context_named_entity = get_ner_in_context('Alexander', document)




