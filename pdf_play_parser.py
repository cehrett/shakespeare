# RUN INSTRUCTIONS:
# To scrape an individual play:
#   <dataframe_name> = play_parser(<title_of_play>)
#   NOTE: This must match the name in plays_to_scrape.csv EXACTLY
# 
# To scrape all plays in plays_to_scrape.csv:
#   scrape_all_plays()
#   NOTE: This function does NOT actually return a csv or dataframe, it simply generates them
#
# Command to turn a dataframe into a csv:
#   <dataframe_name>.to_csv('<csv_name>.csv')



import pip
import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pdfplumber'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'PyPDF2'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'requests'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyarrow'])

# Helper function: add new col to dataframe, telling whether character is 12-pt font
# (Dialogue seems to be always 12 pt)
def twelve_pt_font_col(df):
  twelve_pt_font_mask = abs(df['size'] - 12) < 0.01
  df['twelve_pt_font'] = twelve_pt_font_mask
  return df

  
# Helper function: add new col to dataframe, telling whether character is Times New Roman
# (Dialogue seems to be always TNR)
def times_new_roman_col(df):
  times_mask = ['Times' in font for font in df.fontname.tolist()]
  df['times_new_roman'] = times_mask
  return df
# Helper function: add new col to dataframe, telling whether character is in italics
def italics_col(df):
  import numpy as np
  italics_mask = ['Italic' in font for font in df.fontname.tolist()]
  bracket_mask = [entry in ['[',']'] for entry in df.text]
  combo_mask = np.logical_or(italics_mask, bracket_mask)
  df['italics'] = combo_mask
  return df


# Helper function: add new col to dataframe, telling whether character is in bold
# (Speaker names always appear to be in bold)
def bold_col(df):
  bold_mask = ['Bold' in font for font in df.fontname.tolist()]
  df['bold'] = bold_mask
  return df


# Helper function: Add a column corresponding to the line on the page
def line_no_col(df):
  # This is sensitive to the choice of how far apart two characters need to be, 
  # vertically, to count as being on two different "lines". That choice is made in
  # setting the variable line_diff below -- it may need tinkering.
  line_diff = 4

  import numpy as np
  line = 0 # We'll increment this every time we see a character that is at least line_diff lower on the page
  current_line_val = 0 # We'll update this then too; it will be the vertical position of the character
  line_dict = {}
  for val in np.sort(df.top.unique()):
    if abs(val - current_line_val) > line_diff:
      line += 1
      current_line_val = val
    line_dict[val]=line
  
  # Use the dict to make a new column.
  df['line_number'] = df.top.map(line_dict)
  return df


# function to create labels (dialogue, speaker, direction, etc) based on formatting
def text_type(row):
  # also if in center of page, look at first character of line on page
  # and if there isn't text on same line earlier
  if row['x0'] > 120 and row['fontname'] == 'TimesNewRomanPS-ItalicMT':
    return 'Stage Directions'

  #Good test page: Faustus B pg. 21
  from pandas.core.dtypes.api import is_numeric_dtype
  if row['bold'] == True and row['italics'] == True:
    return 'Speaker'
  
  if row['bold'] == False:
    if row['text'].strip().isnumeric(): 
      return 'Line Number'
    else:
      return 'Dialogue'

#Creates a speaker column that holds who spoke each line of dialogue
def speaker_col(df):
    import numpy as np
    curr_speaker = '-'
    line_dict = {}
    for val in df.index:
      #gets the current speaker (ignores blank lines or lines with only punctuation, but labeled as speaker)
      if df.loc[val]['text_type'] == 'Speaker' and any(s.isalnum() for s in df.loc[val]['text'].strip()):
        curr_speaker = df.loc[val]['text'].strip()

      #Resets the speaker upon a new scene
      if 'SCENE' in df.loc[val]['text']:
        curr_speaker = '-'
      
      #labels who spoke said line of dialogue (excludes lines without actual words to increase count accuracy)
      if (df.loc[val]['text_type'] == 'Dialogue') and any(s.isalnum() for s in df.loc[val]['text'].strip()):
        line_dict[val] = curr_speaker
      else:
        line_dict[val] = '-'

    df['speaker'] = df.index.map(line_dict)
    return df

#Keeps track of the line number of the lines of dialogue in the play
def line_in_play_col(df):
  import numpy as np
  count = 1
  line_dict = {}

  #keeps count of lines of dialogue only (ignores blank lines)
  for val in df.index:
    if (df.loc[val]['text_type'] == 'Dialogue') and any(s.isalnum() for s in df.loc[val]['text'].strip()):
      line_dict[val] = count
      count += 1
    else:
      line_dict[val] = ''
  
  df['line_in_play'] = df.index.map(line_dict)
  return df

#Keeps track of the scene number in the play
def scene_num_col(df):
  import numpy as np
  count = 0
  line_dict = {}

  for val in df.index:
    #if there is a scene change, increase the scene number
    if 'SCENE' in df.loc[val]['text']:
      count += 1

    line_dict[val] = count

  df['scene_num'] = df.index.map(line_dict)
  return df

  # function to fix stage directions
def stage_direction(df):
  #Good test page: Faustus B pg. 21 / pg 44 of df
  import numpy as np

  for val in df.index:
    #This part fixes half of lines being mared as dialogue with the other half being stage directions
    #if a nonempty line is marked as dialogue
    if (df.loc[val]['text_type'] == 'Dialogue') and any(s.isalnum() for s in df.loc[val]['text'].strip()):
      #save its page and line number
      page_num = df.loc[val]['page_number']
      line_num = df.loc[val]['line_number']

      #check all rows with the same page and line number
      for i in df.loc[(df['page_number'] == page_num) & (df['line_number'] == line_num)].index:
        #change its text type to dialogue
        if df.loc[i]['text_type'] == 'Stage Directions':
          df.loc[df.index == i, 'text_type'] = 'Dialogue'

    #This fixes inline stage directions with brackets
    #if a line contains a bracket
    if '[' in df.loc[val]['text']:
      #save its page and line number, and its x0 value
      page_num = df.loc[val]['page_number']
      line_num = df.loc[val]['line_number']
      x0 = df.loc[val]['x0']

      #check all rows with the same page and line number
      for i in df.loc[(df['page_number'] == page_num) & (df['line_number'] == line_num)].index:
        #if it is italicized dialogue that starts close enough to a bracket, change its text type to stage directions
        if (df.loc[i]['text_type'] == 'Dialogue') and (df.loc[i]['x0'] < (x0 + 4)) and (df.loc[i]['italics'] == True):
          df.loc[df.index == i, 'text_type'] = 'Stage Directions'

  return df
      
def scene_correction(df):
  import numpy as np
  for val in df.index:
    if 'SCENE' in df.loc[val]['text']:
      df.loc[df.index == val, 'text_type'] = 'Scene Marker'

    elif (df.loc[val]['text_type'] == 'Dialogue') and (df.loc[val]['speaker'] == '-'):
      df.loc[df.index == val, 'text_type'] = 'Stage Directions'
  return df

def text_stripping(df):
  import numpy as np
  for val in df.index:
    df.loc[df.index == val, 'text'] = df.loc[val]['text'].strip()

  return df

def length_of_line(df):
  import numpy as np
  line_dict = {}

  for val in df.index:
    if any(s.isalnum() for s in df.loc[val]['text']):
      line_dict[val] = len(df.loc[val]['text'])
    else:
      line_dict[val] = 0

  df['length_of_line'] = df.index.map(line_dict)
  return df

def gender_speaker_labels(df, male_list, female_list):
  import numpy as np
  line_dict = {}

  for val in df.index:
    #only label dialogue
    if df.loc[val]['text_type'] == 'Dialogue':
      #classifies male, female, or other gender by whether it is found in the list
      #note: other is being used as a catchall here
      if df.loc[val]['speaker'] in male_list:
        line_dict[val] = 'male'
      elif df.loc[val]['speaker'] in female_list:
        line_dict[val] = 'female'
      else:
        line_dict[val] = 'other'
    else:
      line_dict[val] = '-'

  df['gender_of_speaker'] = df.index.map(line_dict)
  return df

def play_parser(play_name):
  from numpy import number
  from pandas.compat.pyarrow import pa
  import pandas as pd
  import pdfplumber
  import PyPDF2
  import requests

  sheet_df = pd.read_csv('plays_to_scrape.csv')

  #Finds the index of the column with the play values we need of the worksheet
  play_name_index = sheet_df[sheet_df['title']== play_name].index.values
  pdf_link = sheet_df['pdf_link'][play_name_index].values[0]

  # Get the author of the play
  play_author = sheet_df['author'][play_name_index].values[0]

  #Pulls file into a pdf named playfile.pdf
  #All these steps are necesaary for this to work, not sure why
  name = "./playfile.pdf" 
  response = requests.get(pdf_link) # send a GET request to the URL
  file = open(name, "wb") # open a file in write-binary mode
  file.write(response.content) # write the response content to the file
  file.close() # close the file

  #male_list
  male_list_string = str(sheet_df['male_characters'][play_name_index].values[0])
  male_col_list = male_list_string.split(", ")

  #female_list
  female_list_string = str(sheet_df['female_characters'][play_name_index].values[0])
  female_col_list = female_list_string.split(", ")

  #other_list
  other_list_string = str(sheet_df['other_characters'][play_name_index].values[0])
  other_col_list = other_list_string.split(", ")

  start_page = int(sheet_df['start_page'][play_name_index])
  end_page = int(sheet_df['end_page'][play_name_index])

  with open(name, "rb") as pdf_file:
      read_pdf = PyPDF2.PdfReader(pdf_file)
      number_of_pages = len(read_pdf.pages)
      
  
  #with pdfplumber.open(play) as pdf:
  with pdfplumber.open(name) as pdf: 
    pages_list = []

    #Changes the page to end on based on the numer of pages to be omitted
    number_of_pages = number_of_pages - end_page


    #The first few pages are title page info, and so on
    for i in range(start_page, number_of_pages):
        text = pdf.pages[i]
        page_df = pd.DataFrame(text.chars)
        
        # Add columns of useful info
        page_df = times_new_roman_col(page_df)
        page_df = italics_col(page_df)
        page_df = bold_col(page_df)
        page_df = line_no_col(page_df)
        page_df['page_number'] = i + 1
        page_df['text_type'] = page_df.apply (lambda row: text_type(row), axis=1)

        page_df2 = page_df

        # Aggregate data to the level of line on the page
        # Note: fontname and height are needed for the speakers to be properly identified
        page_df1 = page_df.groupby(['line_number', 
        'fontname', 
        'upright', 
        'height',
         'top', 
        'bottom',
        'doctop',
        'times_new_roman', 
        'italics', 
        'bold',
        'page_number',
        'text_type'])['text'].apply(''.join).reset_index()

        page_df2 = page_df.groupby(['line_number', 
        'fontname', 
        'upright', 
        'height',
         'top', 
        'bottom',
        'doctop',
        'times_new_roman', 
        'italics', 
        'bold',
        'page_number',
        'text_type',
        ])['x0'].apply(min).reset_index()

        page_df1['x0'] = page_df2['x0']

        # Add page to list of page dfs
        pages_list.append(page_df1)

  # Concatenate to a single DF
  play_df = pd.concat(pages_list)

  #This gets rid of that line at the very top of the page ('Title Scene #'), which was messing with labeling 
  play_df = play_df[play_df.line_number != 1]

  #resets the index (index was previously labeled per page)
  play_df.reset_index(drop = True, inplace=True)

  #Adds the scene number, labeling of stage directions, speaker column, and corrections
  play_df = scene_num_col(play_df)
  play_df = stage_direction(play_df)
  play_df = speaker_col(play_df)
  play_df = scene_correction(play_df)
  play_df = text_stripping(play_df)

  # Remove empty text rows and brackets
  play_df = play_df[play_df.text != ' ']
  play_df = play_df[play_df.text != '[]']

  #Keeps relevant data
  play_df = play_df.groupby([
      'page_number',
      'scene_num',
      'line_number',
      'text_type',
      'speaker'
        ])['text'].apply(''.join).reset_index()

  #Add length of each line, as well as the line in the overall play
  play_df = length_of_line(play_df)
  play_df = play_df[play_df.length_of_line != 0]
  play_df = line_in_play_col(play_df)
  play_df.reset_index(drop = True, inplace=True)

  #Adds gender of speaker labels
  play_df = gender_speaker_labels(play_df, male_col_list, female_col_list)

  # Add play name to play_df
  play_df['title'] = play_name

  # Add play author to play_df
  play_df['author'] = play_author

  return play_df

def sort_by_paragraph(df):
  import numpy as np
  import pandas as pd

  #indexing for the new dataframe (this is set to -1 so that indexing will start at 0)
  new_index = -1
  curr_speaker = ""
  line_dict_text = {}
  line_dict_speaker = {}
  line_dict_gender = {}
  line_dict_scene_num = {}

  for val in df.index:
    #checks if this is a line of dialogue with a valid speaker
    if (df.loc[val]['text_type'] == 'Dialogue') and (df.loc[val]['speaker'] != '-'):
      #if the speaker is the same, we're adding to the same index
      if df.loc[val]['speaker'] == curr_speaker:
        line_dict_text[new_index] += " " + df.loc[val]['text']

      #otherwise, store the new speaker and switch to a different list
      else:
        new_index += 1
        curr_speaker = df.loc[val]['speaker']
        line_dict_text[new_index] = df.loc[val]['text']
        line_dict_speaker[new_index] = curr_speaker
        line_dict_gender[new_index] = df.loc[val]['gender_of_speaker']
        line_dict_scene_num[new_index] = df.loc[val]['scene_num']

  #Stroes all of this info into the new dataframe and returns it
  paragraph_df = pd.Series(line_dict_text).to_frame('text')
  paragraph_df['speaker'] = paragraph_df.index.map(line_dict_speaker)
  paragraph_df['gender_of_speaker'] = paragraph_df.index.map(line_dict_gender)
  paragraph_df['scene_num'] = paragraph_df.index.map(line_dict_scene_num)
  paragraph_df = length_of_line(paragraph_df)

  return paragraph_df

def scrape_all_plays():
  import pandas as pd
  sheet_df = pd.read_csv('plays_to_scrape.csv')
  num_of_plays = len(sheet_df)

  for i in range(num_of_plays):
    play_name = str(sheet_df['title'][i])
    start_page = str(sheet_df['start_page'][i])
    end_page = str(sheet_df['end_page'][i])

    df = play_parser(int(start_page), int(end_page), play_name)
    df_name = play_name.replace(' ', '_')
    df_name = df_name.replace(',', '')
    df_name = df_name.replace("'", '')
    df_name = df_name.replace("’", '')
    df_name = df_name + "_df.csv"

    df.to_csv(df_name)

#FOR TEST PURPOSES
#df = play_parser("The Massacre at Paris")
#df = play_parser("Tamburlaine the Great, Part One")

#df.to_csv('play_df.csv')
#scrape_all_plays()
