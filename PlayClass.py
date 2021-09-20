# This script implements the Play class, which scrapes a webpage containing a play. It contains a method for
#   gathering Watson NLU metadata on the play. It also contains methods for
#   plotting the WNLU metadata, and for exporting the play contents and metadata
#   as a pandas dataframe.

# Any needed imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from string import digits
from bs4 import BeautifulSoup, NavigableString
import json
from concurrent.futures import ThreadPoolExecutor
from ibm_watson import NaturalLanguageUnderstandingV1 as NLU #<-- import NLU
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions, CategoriesOptions, SentimentOptions
features=Features(
    sentiment=SentimentOptions(),
    emotion=EmotionOptions()
)
import seaborn as sns


def remove_tags(soup):
    """
    # Removes unecessary tags (and their content)
    """
    for span in soup.find_all('span', {'class': 'splitline-indent'}):
        span.decompose()
    for qln in soup.find_all('a', {'id': re.compile('^qln')}):
        qln.decompose()
    for link in soup.find_all('a', {'id': re.compile('^l-')}):
        link.decompose()
    for invis in soup.find_all('a', {
        'class': ['milestone qadd invisible', 'milestone fadd invisible', 'milestone prose invisible',
                  'milestone FM invisible',
                  'milestone Johnson invisible', 'milestone q1 invisible']}):
        invis.decompose()

    # This gets rid of span tags but keeps the content of those span tags
    invalid_tags = ['span']
    for tag in invalid_tags:
        for match in soup.findAll(tag):
            match.replaceWithChildren()


regtln = re.compile('^tln')


def is_tln(tag):
    """
    This function checks whether a BS4 item is a tln tag
    """
    if tag.name:
        if tag.has_attr('id'):
            return ((tag.name == 'a') & bool(regtln.match(tag['id'])))

    # Else:
    return False


def proc_div(line):
    """
    The first function, proc_div, takes as input a div line from the html for the
    page, and outputs a list of dictionaries corresponding to lines from the play.
    Each dictionary is either a TLN number, or else a bit of dialogue, or else a
    speaker name or a scene name.
    """

    # The dictionary is used to get line type from the class of a div.
    typedict = {'sponly': 'speaker', 'sdonly': 'stage_direction', 'line': 'line', 'ldonly': 'line_direction',
                'center': 'centered line', 'right': 'right_align'}
    list_of_dicts = []

    # Get type of line
    try:
        if line['class'][-1] in ['interrupt', '']:
            divtype = line['class'][-2]
        else:
            divtype = typedict[line['class'][-1]]

    except:
        print(line['class'])
        print(line)
        raise Exception("Error getting div type.")

    if divtype == 'sdonly':
        divtype = 'stage_direction'
    elif divtype == 'sponly':
        divtype = 'speaker'
    divcontents = line.contents

    scene = line.find("span", class_="sceneleader")
    if scene is not None:
        scenes = scene.text
    else:
        scenes = None

    speak = line.find("span", class_="speaker")
    if speak is not None:
        speaks = speak.text
    else:
        speaks = None

    while divcontents:
        dat = {}

        if is_tln(divcontents[0]):
            dat['type'] = 'TLN'
            dat['tln'] = divcontents.pop(0).text
            list_of_dicts.append(dat)
        elif str(divcontents[0]) == ' ':
            divcontents.pop(0)
        else:
            divtext, divcontents = proc_list_of_div_children(divcontents)
            dat['type'] = divtype
            dat['text'] = divtext

            # catches if anything is left in between <> tags or if any digits appear in text
            remove_digits = str.maketrans('', '', digits)
            remove_chars = set('<>')
            remove_chars2 = set('>')

            if any((c in remove_chars) for c in divtext):
                divtext_temp = re.sub(r'<.+?>', '', divtext)
                if any((c in remove_chars2) for c in divtext_temp):
                    divtext_temp = divtext_temp.split('>')[1]

                divtext_clean = divtext_temp.translate(remove_digits)
                divtext_clean = " ".join(divtext_clean)

                # replaces special characters that appear after webscraping
                divtext_clean = divtext_clean.replace('√®', 'e')

                dat['text_cleaned'] = divtext_clean
            else:
                divtext_clean = divtext.translate(remove_digits)
                divtext_clean = divtext_clean.replace('√®', 'e')

                dat['text_cleaned'] = divtext_clean

            if divtype is 'speaker':
                dat['speaker'] = divtext
                proc_div.scount += 1
                dat['text_block'] = proc_div.scount
                dat['text'] = None
                dat['text_cleaned'] = None

            if divtype is 'line_direction':
                dat['scene'] = divtext
                dat['text'] = None
                dat['text_cleaned'] = None

            list_of_dicts.append(dat)

    return list_of_dicts


proc_div.scount = 0


def proc_list_of_div_children(lodc):
    """
    The function proc_list_of_div_children is called within proc_div. This function
    takes as input a list of contents from a div line and joins together the text
    from consecutive elements of that list until it hits either (1) the end, or
    (2) a tln. Then it outputs the string that is the text from the elements it
    joined together, along with the remainder of the list.
    """
    linetext = ""
    while lodc and not is_tln(lodc[0]):
        linetext += str(lodc.pop(0))
    return linetext, lodc


def proc_div_mit(line):
    """
    proc_div for scraping from MIT website takes as input a div line from the html for the
    page, and outputs a list of dictionaries corresponding to lines from the play.
    Each dictionary is either a stage direction, or else a bit of dialogue, or else a
    speaker name or a scene name.
    """

    dat = {}

    # act and scene names
    if (line.name == 'h3'):
        linetype = 'act'
    # stage dir
    if (line.name == 'i'):
        linetype = 'stage_direction'
    # speaker or line text
    if (line.name == 'a'):
        if (re.search("^speech", line['name'])):
            linetype = 'speaker'
        # class name is a line number (ex: 1.3.35)
        elif (re.search("^\d+(\.\d+)*$", line['name'])):
            linetype = 'line'
            linenumber = line['name']

    # creating dict with text data for different html line types
    if linetype is 'act':
        dat['act'] = line.text
        dat['speaker'] = None
        dat['text'] = None
        dat['type'] = linetype


    elif linetype is 'stage_direction':
        dat['text'] = line.text
        dat['speaker'] = None
        dat['act'] = None
        dat['type'] = linetype

    elif linetype is 'line':
        dat['text'] = line.text
        dat['type'] = linetype
        dat['speaker'] = None
        dat['act'] = None
        dat['line_number'] = linenumber

    elif linetype is 'speaker':
        dat['speaker'] = line.text
        dat['type'] = linetype
        dat['text'] = None
        dat['act'] = None

    return dat


def find_scene(df, tln):
    """
    Finds scene of the closest tln to a line from an dataframe input
    """
    mintln = df['tln'].iloc[0]
    try:
        scene = df.loc[df['tln'] == tln, 'scene'].iloc[0]
    # if this tln doesn't exist, find the scene of the nearest tln
    except:
        if tln < mintln:
            scene = df.loc[df['tln'] == mintln, 'scene'].iloc[0]
        else:
            scene_idx = ((df['tln'] - tln) ** 2).idxmin()
            scene = df.loc[scene_idx, 'scene']
    return scene


def get_window_text(df_start, window):
    """
    Combines tlns into a a rolling window of text. Takes an input dataframe and
    concatenates a window of tlns into one string.
    """
    list_tlns = []

    endloop = int(df_start['tln'].max() + 1)

    for i in range(endloop):
        dats = {}
        tlnstart = i - (window / 2)
        tlnend = i + (window / 2)
        # isolate window of tlns text
        df_window = df_start[(df_start['tln'] >= tlnstart) & (df_start['tln'] <= tlnend)]
        text_list = df_window.CONTENT

        # joining rows of strings and removing any double spaces
        join_str = [' '.join(str(idx) for idx in text_list)]
        temp_str = join_str[0]
        line_nospace = " ".join(temp_str.split())

        window_idx = i + 1

        if tlnstart < 1:
            tlnstart = 0
        if tlnend > df_start['tln'].max():
            tlnend = df_start['tln'].max()

        dats['tln_start'] = tlnstart
        dats['tln_end'] = tlnend

        dats['window'] = window_idx

        scene_start = find_scene(df_start, tlnstart)
        scene_end = find_scene(df_start, tlnend)

        dats['scene'] = scene_start
        dats['scene_end'] = scene_end

        dats['CONTENT'] = line_nospace

        list_tlns.append(dats)

    return list_tlns


def get_ticks(df):
    """
     Finding tlns where scenes change for plot tick marks
    """
    changes = {}
    for col in df.columns:
        changes[col] = [0] + [idx for idx, (i, j) in enumerate(zip(df[col], df[col][1:]), 1) if i != j]

    scenechanges = (changes['scene'])
    scenetlns = []
    scenes = []
    # getting lists of scenes and tlns
    for i in scenechanges:
        scenetlns.append(df.tln.iloc[i])
        scenes.append(df.scene.iloc[i])
    scenetlns = pd.Series(scenetlns).fillna(0).astype(float).tolist()
    # replacing every other element with '' to avoid clutter on graph
    scenes[1::2] = ['' for x in scenes[1::2]]
    xticks = {'tln_ticks': scenetlns, 'scene_ticks': scenes}

    return xticks


def get_response(x, nlu):
    """
     Get response from watson NLU
    """
    ## NOTE - Currently the code is set up to use the column "CONTENT" of the line, with Content being the text ##
    try:
        resp = nlu.analyze(text=x.CONTENT,
                           features=Features(
                               sentiment=SentimentOptions(),  # Get sentiment  (1 token)
                               emotion=EmotionOptions())  # Get the 5 emotion responses (1 token)
                           ).get_result()
    except Exception as e:
        resp = None
        print("Something went wrong.... Error: ", e)

    return resp


def condition_response(resp):
    """
    Prepare the response for entry into a dataframe
    Shouldn't have to change these if you're using both sentiment and emotion
    """
    if resp is None:
        cr = dict(nlu_status="fail",
                  emotion_anger=None,
                  emotion_disgust=None,
                  emotion_fear=None,
                  emotion_joy=None,
                  emotion_sadness=None,
                  sentiment_label=None,
                  sentiment_score=None,
                  language=None,
                  n_char=None)
    else:
        emotions = None if resp.get('emotion') is None else resp['emotion']['document']['emotion']
        sentiments = None if resp.get('sentiment') is None else resp['sentiment']['document']

        cr = dict(nlu_status="pass",
                  emotion_anger=None if emotions is None else emotions['anger'],
                  emotion_disgust=None if emotions is None else emotions['disgust'],
                  emotion_fear=None if emotions is None else emotions['fear'],
                  emotion_joy=None if emotions is None else emotions['joy'],
                  emotion_sadness=None if emotions is None else emotions['sadness'],
                  sentiment_label=None if sentiments is None else sentiments['label'],
                  sentiment_score=None if sentiments is None else sentiments['score'],
                  language=resp.get('language'),
                  n_char=None if resp.get('usage') is None else resp['usage']['text_characters'])

    return cr


def process_row(x, nlu, window):
    """
    Process an individual row or windowed row
    """
    # Columns for a single tln
    if window == 1:
        cr = condition_response(get_response(x, nlu))
        cr['tln'] = x['tln']
        cr['CONTENT'] = x['CONTENT']
        cr['type'] = x['type']
        cr['scene'] = x['scene']
        cr['speaker'] = x['speaker']
        cr['year'] = x['year']
    # Columns for a window of tlns
    else:
        cr = condition_response(get_response(x, nlu))
        cr['tln_start'] = x['tln_start']
        cr['tln_end'] = x['tln_end']
        cr['window'] = x['window']
        cr['scene'] = x['scene']
        cr['scene_end'] = x['scene_end']
        cr['CONTENT'] = x['CONTENT']

    return cr


def get_wnlu_results(df, wnlu_credentials, window, nthread=20):  # 30 max, 20 to be safe, 40 to be reckless

    """
    Gets WNLU results by row and accesses WNLU credentials
    """

    # Accessing wnlu credentials
    with open(wnlu_credentials) as f:
        creds_dict = json.load(f)
    nlu = NLU(
        version='2020-02-19',
        iam_apikey=creds_dict["apikey"],
        url=creds_dict["url"]
    )

    # get an iterable for the rows of the dataframe
    rows = df.iterrows()
    nrows = df.shape[0]

    # wrapper function to handle extra index in iterable
    def process_row_no_idx(row):
        return process_row(row[1], nlu, window)

    # send rows to Watson NLU asynchronously
    with ThreadPoolExecutor(max_workers=nthread) as executor:
        futures = executor.map(process_row_no_idx, rows)

    # turn above iterator into list of results
    list_results = []
    for r in futures:
        list_results.append(r)

    # return the results
    return list_results


def plot_moving_average(series, window, ticks, plot_intervals=False,
                        scale=1.96, center=True, plot_actual=False,
                        label=None, title=None, ax=None, dropna=False):
    """
    This function plots a rolling average (where the window is defined over tlns)
    """
    if not ax:
        print('Creating axes')
        fig = plt.figure(figsize=(17, 8))  # create a figure object
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure

    if dropna:
        rolling_mean = series.dropna().rolling(window=window, center=center).mean()
    else:
        rolling_mean = series.rolling(window=window, center=center, min_periods=1).mean()

    # plt.figure(figsize=(17,8))
    if title:
        plt.title(title + '\n window size = {}'.format(window))
    plt.plot(rolling_mean, linewidth=3, alpha=.75)

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound', ax=ax)
        plt.plot(lower_bound, 'r--', ax=ax)

    if plot_actual:
        plt.plot(series[window:], label='Actual values', ax=ax)

    plt.grid(True, axis='x', alpha=.25)
    plt.xticks(ticks['tln_ticks'],
               ticks['scene_ticks'])

    return ax


def rolling_average_comparison(df, ax, score_type: str, window: int, character: str, ticks: dict):
    """
    This function takes in a df and a 'score type' (e.g. anger) and feeds it to plot_moving_average.
    """

    # Get relevant columns:
    score_columns = [col for col in df.columns if score_type.lower() in col]

    title = character + '\'s ' + score_type.capitalize()

    # Make the plots
    ax = plot_moving_average(df[score_columns[0]], window, title=title, ax=ax, ticks=ticks)
def make_speaker_list(df):
  """
  This function returns a list of every unique speaker in a df
  """

  speaker_list = df.speaker.unique().tolist()
  speaker_list = [word.upper() for word in speaker_list]

  return speaker_list


def stage_finder(x, speaker_list):
  """
  This function scans a stage direction line  (Enter Othello, Exit Macbeth ...) and finds every 
  character present in that line. This keeps track of characters who enter or exit each line.
  Returns a df row with 3 new columns for characters who enter/exit, or if everyone exits (exeunt in the play)
  """

  enter_string =x['stage'].partition("Enter ")[2] or x['stage'].partition("Re-enter ")[2]
  enter_list = [i for i in speaker_list if i in enter_string.upper()]
  while("" in enter_list): 
    enter_list.remove("")

  x['Enter'] = enter_list

  exit_string = x['stage'].partition("Exit ")[2]
  exit_list = [i for i in speaker_list if i in exit_string.upper()]
  while("" in exit_list): 
    exit_list.remove("")
  # Removes things like "MESSENGER" if "SECOND MESSENGER"
  x['Exit'] = exit_list

  exeunt_string = x['stage'].partition("Exeunt")[2] or x['stage'].partition("Exeunt.")[2]
  exeunt_list = [i for i in speaker_list if i in exeunt_string.upper()]
  if ((len(exeunt_list) == 1) and (exeunt_list[0] == '') and (str(x.text).startswith("Exeunt"))): # and (str(x.CONTENT).startswith("Exeunt"))):
    exeunt_list[0] = 'ALL'


  x['Exeunt'] = exeunt_list

  return x

def finding_audience(df):
  """
  This function finds each character that is present during each line in the df.
  Uses stage_finder function then iterates through each line to create a list 
  which keeps track of current characters in the scene.
  """

  stagedir = df.copy()
  stagedir['stage'] = ''
  stagedir['stage'] = np.where((stagedir['type'] == 'stage_direction'), #Identifies the case to apply to
                            stagedir['text'],      #This is the value that is inserted #used to be stagedir['CONTENT'] but text has info in []
                            stagedir['stage'])      #This is the column that is affected
                          
  stagedir.stage = stagedir.stage.fillna('')


  stagedir = stagedir.apply(stage_finder, speaker_list= make_speaker_list(df), axis=1)

  stagedir['Present'] = np.empty((len(stagedir), 0)).tolist()

  present_now = []
  for index, row in stagedir.iterrows():
    # empties current df column
    row['Present'].clear()

    # Checks Enter column and appends to present list
    if len(row['Enter']) > 0:
      for char in row['Enter']:
        if char not in present_now:
          present_now.append(char) 
    # Checks exit column and removes any characters from present list
    if len(row['Exit']) > 0:
      for char in row["Exit"]:
        try:
          present_now.remove(char)
        except:
          pass
    # special case where text is only "Exit", removes current speaker
    if (row['stage'] == 'Exit' or row['stage'] == 'Exit.' ) & (row['type'] == 'stage_direction'):
      present_now.remove(row['speaker'])

    # Exeunt column    
    if len(row['Exeunt']) > 0:
      for char in row["Exeunt"]:
        # Removes all current characters in present list
        if char == 'ALL':
          present_now.clear()
        # Only removes specific chars
        else:
          try:
            present_now.remove(char)
          except:
            pass
    # Tries to catch when Enter column misses some characters, inputs any subsequent speakers that appear
    if (row['speaker'] not in present_now) and (row['type'] == 'line'):
      present_now.append(row['speaker'])

    # adding current list iteration to df
    present_now = list(set(present_now))
    for word in present_now:  
      stagedir.at[index,'Present'].append(word) 

  return stagedir

def isolate_text(df,df_with_only_lines):
  """
  df - A dataframe which has the current audience (probably result of finding_audience fxn)
  df_with_only_lines - Dataframe with WNLU results, no audience column

  This function will merge these dataframes to only keep the lines (adding audience data to WNLU dataframe)
  """
  try:
    df_with_only_lines = df_with_only_lines.rename(columns={"text": "CONTENT"})
  except:
    pass

  merged = pd.merge(df, df_with_only_lines, how='inner',on=['CONTENT'])

  merged.drop(merged.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
  merged = merged.loc[:, ~merged.columns.str.contains('^Unnamed')]

  return merged


def check_for_character(x, character_list):
  """
  This function checks if a row has any character present that is also in 
  a specified list of either outsiders or nobility. 
  """
  current_list = x.Present
  character_list = [word.upper() for word in character_list]
  check = any(item in current_list for item in character_list)
  return check

# Class definition

class Play:
    """
    This class scrapes a webpage containing a play. It contains a method for
    gathering Watson NLU metadata on the play. It also contains methods for
    plotting the WNLU metadata, and for exporting the play contents and metadata
    as a pandas dataframe.
    """

    def __init__(self,
                 url: str = None,
                 filepath: str = None,
                 wnlu_credentials: str = None):
        self.url = url
        self.filepath = filepath
        self.wnlu_credentials = wnlu_credentials

    def scrape(self, url: str = None, year: int = 1599):
        """
        This function uses self.url to scrape the play from that url.
        The year input is the estimated year the play was written
        """
        # Define inputs
        if url == None: url = self.url

        # Requests html page from url
        page = requests.get(url)
        # Uses beautiful soup to extract html elements
        soup = BeautifulSoup(page.text, 'html.parser')

        remove_tags(soup)

        # Extracts html with class 'line'
        lines = soup.find_all('div', {'class': ['line']})

        all_data = []
        for line in lines:
            line_dicts = proc_div(line)
            all_data.extend(line_dicts)

        # Creating pandas dataframe from all_data dict
        play_df = pd.DataFrame(all_data)
        # removing anything in between []
        play_df.text_cleaned = play_df.text_cleaned.str.replace("\[.*?\]", '')
        # removing any extra spaces in text strings
        play_df.text_cleaned = play_df.text_cleaned.replace('\s+', ' ', regex=True)
        play_df.text_cleaned = play_df.text_cleaned.str.replace('^ +', '')
        play_df.speaker = play_df.speaker.str.replace('^ +', '')
        play_df.speaker = play_df.speaker.str.replace(' +$', '')

        # removing 'Exeunt' from lines
        play_df['CONTENT'] = play_df['text_cleaned'].str.split('Exeunt').str[0]

        # adding year that the play was written (rough estimate)
        play_df['year'] = year

        # Formatting column order
        new_cols = ['year', 'type', 'scene', 'speaker', 'tln', 'text', 'CONTENT', 'text_block']
        play_df = play_df[new_cols]

        # Forward filling columns
        play_df.speaker = play_df.speaker.ffill()
        play_df.tln = play_df.tln.ffill()
        play_df.scene = play_df.scene.ffill()
        play_df.text_block = play_df.text_block.ffill()

        self.play_df = play_df

    def scrape_mit(self, url: str = None, year: int = 1596):
        """
        This function uses self.url to scrape the play from that url.
        The year input is the estimated year the play was written
        """
        # Define inputs
        if url == None: url = self.url

        # Requests html page from url
        page = requests.get(url)
        # Uses beautiful soup to extract html elements
        soup = BeautifulSoup(page.text, 'html.parser')
        html = list(soup.children)[2]
        body = list(html.children)[3]

        # Remove tags we don't need (and their content)
        for href in body.find_all('a', href=True):
            href.decompose()

        # Getting data in a dict
        all_data = []
        tln = 1
        for line in body.find_all(['h3', 'a', 'i']):
            line_dicts = proc_div_mit(line)
            if line_dicts['type'] == 'line':
                line_dicts['tln'] = tln
                tln = tln + 1
            else:
                line_dicts['tln'] = None

            all_data.append(line_dicts)

        play_df = pd.DataFrame(all_data)
        play_df['CONTENT'] = play_df['text']
        # removing anything in between []
        play_df.CONTENT = play_df.CONTENT.str.replace("\[.*?\]", '')
        # removing any extra spaces in text strings
        play_df.CONTENT = play_df.CONTENT.replace('\s+', ' ', regex=True)
        play_df.CONTENT = play_df.CONTENT.str.replace('^ +', '')
        play_df.speaker = play_df.speaker.str.replace('^ +', '')
        play_df.speaker = play_df.speaker.str.replace(' +$', '')

        play_df.speaker = play_df.speaker.str.capitalize()

        # removes extra text in act/scene lines
        play_df['scene_temp'] = play_df['act'].str.split('.').str[0]
        play_df.scene_temp = play_df.scene_temp.fillna('')
        # getting mask for when act changes
        play_df['act'] = list(
            map(lambda x: x.startswith('ACT'), play_df['scene_temp']))

        # forward fill speaker
        play_df.speaker = play_df.speaker.ffill()

        # forward fill tln
        play_df.tln = play_df.tln.ffill()
        play_df.line_number = play_df.line_number.ffill()

        # '' to Nan
        play_df.scene_temp = play_df.scene_temp.replace(r'^\s*$', np.nan, regex=True)

        # forward fill scene
        play_df.scene_temp = play_df.scene_temp.ffill()

        # applies mask to get act on each row
        play_df['act2'] = play_df['scene_temp'][play_df['act']]
        play_df.act2 = play_df.act2.ffill()

        # combines act and scene numbers into one last column
        play_df["scene"] = play_df["act2"].astype(str) + '. ' + play_df["scene_temp"].astype(str)

        play_df['year'] = year
        new_cols = ['year', 'type', 'scene', 'speaker', 'tln', 'line_number', 'text', "CONTENT"]
        play_df = play_df[new_cols]

        self.play_df = play_df

    def get_wnlu(self,
                 window: int = 1,
                 mask: pd.Series = None,
                 wnlu_credentials=None):
        """
        This function gathers Watson NLU metadata on self.playtext, chunking it
        into bits of size window. Optionally, mask can be used to gather data on
        only part of the play; e.g. if we wish to gather metadata for only one
        character.
        """
        # Define inputs
        if wnlu_credentials == None: wnlu_credentials = self.wnlu_credentials
        play_df = self.play_df
        # gets dataframe of only lines with text (no stage/line direction or speaker types)
        lines_only = play_df[play_df['type'] == 'line']
        if mask is not None:
            lines_only = lines_only[mask]
        # Getting wnlu results for each line by itself (window = 1)
        if window <= 1:
            play_data = get_wnlu_results(lines_only, wnlu_credentials, window, nthread=20)
            # turn the list into a dataframe
            play_df = pd.DataFrame(play_data, columns=['CONTENT', 'tln', 'type', 'scene', 'speaker', 'year',
                                                       'nlu_status',
                                                       'language',
                                                       'n_char',
                                                       'sentiment_label',
                                                       'sentiment_score',
                                                       'emotion_anger',
                                                       'emotion_disgust',
                                                       'emotion_fear',
                                                       'emotion_joy',
                                                       'emotion_sadness'])
        elif window > 1:  # wnlu results for any window
            # making tln columns into floats
            lines_only['tln'] = lines_only['tln'].astype(float)
            # getting windowed dataframe of lines
            windowed_data = get_window_text(lines_only, window)
            window_df = pd.DataFrame(windowed_data)

            play_data = get_wnlu_results(window_df, wnlu_credentials, window, nthread=20)
            # turn the list into a dataframe
            play_df = pd.DataFrame(play_data,
                                   columns=['tln_start', 'tln_end', 'scene', 'scene_end', 'CONTENT', 'window',
                                            'nlu_status',
                                            'language',
                                            'n_char',
                                            'sentiment_label',
                                            'sentiment_score',
                                            'emotion_anger',
                                            'emotion_disgust',
                                            'emotion_fear',
                                            'emotion_joy',
                                            'emotion_sadness'])

        self.play_df = play_df

    def line_plot(self,
                  smooth_window: int = 1,
                  graph_title: str = None,
                  mask: pd.Series = None):
        """
        This function uses the stored WNLU metadata to make a line plot. The line
        plot can be smoothed with smooth_window, and can be restricted in any way
        one likes using mask.
        """

        if graph_title == None: graph_title = self.graph_title
        play_df = self.play_df

        if mask is not None:
            play_df = play_df[mask]

        # Getting lists for plotting tick marks
        xticks = get_ticks(play_df)

        # Drop rows with no WNLU data
        play_df = play_df[play_df['nlu_status'] == 'pass']

        # Rename columns
        rename_map = {'CONTENT': 'text', 'sentiment_score': 'sentiment', 'emotion_anger': 'anger',
                      'emotion_disgust': 'disgust', 'emotion_fear': 'fear', 'emotion_joy': 'joy',
                      'emotion_sadness': 'sadness'}
        play_df.rename(columns=rename_map, inplace=True)

        # Aggregate individual tlns that have same speaker. Concatenate line text, and mean the sentiment/emotion scores. About 200 of these lines.
        aggregation_map = {'text': ' '.join, 'sentiment': np.mean, 'anger': np.mean, 'disgust': np.mean,
                           'fear': np.mean, 'joy': np.mean, 'sadness': np.mean}
        play_df = play_df.groupby(['tln', 'speaker', 'scene', 'year'], as_index=False).agg(aggregation_map)

        # Remove square brackets from some speaker strings
        play_df['speaker'] = play_df['speaker'].str.replace('[', '').str.replace(']', '')
        # Remove whitespace from speaker names
        play_df['speaker'] = play_df['speaker'].str.strip()

        # Set index to be tln NOTE THAT THIS IS IMPERFECT DUE TO SOME TLNS HAVING MULTIPLE "LINES" WITH DIFFERENT SPEAKERS
        play_df.set_index(['tln'], inplace=True)

        fig = plt.figure(figsize=(17, 12))

        score_types = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'sentiment']

        # Plotting line graphs for each emotion/sentiment score
        for score_type, num in zip(score_types, range(1, 7)):
            ax = fig.add_subplot(3, 2, num)
            rolling_average_comparison(play_df, ax=ax, score_type=score_type, window=smooth_window,
                                       character=graph_title, ticks=xticks)
            sns.despine(right=True, left=True, bottom=True)
            plt.setp(ax.get_xticklabels(), rotation=50, horizontalalignment='right')

        # Adjust spacing
        plt.subplots_adjust(hspace=.45)

        plt.savefig(graph_title + '_scores.png')

    def box_plot(self,
                 graph_title: str = None,
                 speaker_list: list = [''],
                 mask: pd.Series = None):
        """
        This function uses the stored WNLU metadata to make a box plot. The box plot can have boxes for each
        score type (no input speaker list) or boxes for a specified list of speakers (speaker list input),
        and can be restricted in any way one likes using mask.
        """

        # Make box plots with CIs for the standard error of the mean scores
        if graph_title == None: graph_title = self.graph_title
        play_df = self.play_df

        if mask is not None:
            play_df = play_df[mask]

        # Rename columns
        rename_map = {'CONTENT': 'text', 'sentiment_score': 'sentiment', 'emotion_anger': 'anger',
                      'emotion_disgust': 'disgust', 'emotion_fear': 'fear', 'emotion_joy': 'joy',
                      'emotion_sadness': 'sadness'}
        play_df.rename(columns=rename_map, inplace=True)

        # Plots one boxplot with each Watson NLU Score
        if len(speaker_list) == 1:

            # adds columns needed to plot NLU scores on same plot
            for nlu_type in ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Sentiment']:
                play_df[str(nlu_type)] = str(nlu_type)

            # Make box plots with CIs for the standard error of the mean scores
            fig = plt.figure(figsize=(17, 12))

            score_types = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'sentiment']
            score_labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Sentiment']

            # Plotting each score type
            for score_type, num in zip(score_types, range(1, 7)):
                score_label = score_type.capitalize()
                g = sns.boxplot(x=score_label, y=score_type, data=play_df, order=score_labels, palette='muted')
                g.set(xlabel='')
                g.set(ylabel='NLU Score')
                g.set(title=graph_title)

        # Plotting 6 boxplots limited to only certain speakers
        elif len(speaker_list) > 1:
            speakers_df = play_df[[speaker in speaker_list for speaker in play_df['speaker']]]
            # Make boxplots
            fig = plt.figure(figsize=(18, 12))

            score_types = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'sentiment']

            for score_type, num in zip(score_types, range(1, 7)):
                ax = fig.add_subplot(3, 2, num)
                g = sns.boxplot(x="speaker", y=score_type, data=speakers_df, ax=ax)
                g.set(xlabel='')
                g.set(ylabel=score_type.capitalize())

            # Adjust spacing
            plt.subplots_adjust(hspace=.35)

        plt.savefig(graph_title + '_boxplot.png')

    def export_data(self,
                    mask: pd.Series = None,
                    export_filepath: str = None):
        """
        This function exports the stored data to a csv.
        """
        if export_filepath == None: export_filepath = './play_df.csv'
        self.play_df.to_csv(export_filepath)

    def character_mask(self,
                       character_list: list):
        """
        This function creates a mask (a Boolean series the same length as
        self.play_df) which is true if and only if the relevant line is spoken by
        a character in character_list.
        """

        # Define inputs
        play_df = self.play_df

        # gets dataframe of only lines with text (no stage/line direction or speaker types)
        play_df = play_df[play_df['type'] == 'line']
        # Remove square brackets from some speaker strings
        play_df['speaker'] = play_df['speaker'].str.replace('[', '').str.replace(']', '')
        # Remove whitespace from speaker names
        play_df['speaker'] = play_df['speaker'].str.strip()

        mask_character = play_df['speaker'].isin(character_list)

        self.mask_character = mask_character

    def add_outsider_label(self,
                           input_csv: str = None,
                           type_dictionary: dict = None,
                           female_list: list = None,
                           gender_ambivalent: list = None):

        """
        This function takes in a df and a dict with speakers and an associated outsider type (e.g. race)
        and adds a column with these outsider types for each line spoken by the speakers in the dict.
        """
        # Define inputs

        df = pd.read_csv(input_csv)

        play_df = df
        play_df['speaker'] = play_df['speaker'].replace('\s+', ' ', regex=True)

        # creating outsider binary label
        play_df['outsider'] = np.where(play_df['speaker'].isin(list(type_dictionary.keys())), 1, 0)
        # mapping type of outsider to a new column
        play_df['outsider_type'] = play_df['speaker'].map(type_dictionary)
        play_df = play_df.drop(columns='outsider')

        ####
        if female_list is not None:
            play_df['gender'] = np.where(play_df['speaker'].isin(female_list), 'female', 'male')

        if gender_ambivalent is not None:
            play_df['gender'] = np.where(play_df['speaker'].isin(gender_ambivalent), 'gender ambivalent',
                                         play_df['gender'])
        try:
            play_df = play_df.drop(columns=['Disgust', 'Anger', 'Joy', 'Sadness', 'Fear', 'Sentiment'])
        except:
            print('no extra cols')

        self.play_df = play_df
        
    def find_group_presence(self, character_list, raw_df, check_for_outsiders = True, lines_df = None):
        """
        character_list - list with all outsiders OR nobility in the play
        check_for_outsiders - Should be True if adding an Outsider_Present column
                            False for Nobility_Present column
        raw_df - This is a dataframe without NLU data, it should contain the result from web-scraping
        lines_df - This is a dataframe with NLU data for each line of a play, if this is not provided
                The function will use self.play_df (Therefore, self.get_wnlu() should have been called already)

        Will set self.play_df to the updated df

        """
        if lines_df is None:
            lines_df = self.play_df

        try:
            self.lines_df = self.lines_df.rename(columns={"text": "CONTENT"})
        except:
            pass

        df = raw_df.copy()
        df.speaker = df.speaker.fillna('')
        df.speaker = df.speaker.apply(lambda x: x.upper())
        df = df.rename(columns={"text_cleaned": "CONTENT"})
        df.speaker = df.speaker.astype(str)
        df = finding_audience(df)

        df = isolate_text(df, lines_df)

        if not check_for_outsiders:
            df['class'] = np.where(df['speaker_x'].isin(character_list), 'nobility', 'servant')

        if check_for_outsiders:
            df['Outsider_Present'] = df.apply(lambda x: check_for_character(x,character_list), axis=1)
        else:
            df['Nobility_Present'] = df.apply(lambda x: check_for_character(x,character_list), axis=1)


        self.play_df = df


