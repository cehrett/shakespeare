from langchain.prompts import PromptTemplate

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

PROMPT = """Given the title of a play, its published year, author, genre, and three lines, generate the next line that follows.
Title: {title}
Year: {year}
Author: {author}
Genre: {genre}
Lines of play: 
{lines_of_play}

Next line:
"""

DEFAULT_SYSTEM_PROMPT = """You are an expert at predicting the next line of Elizabethan plays."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


def create_prompt(title):
    import pandas as pd

    play_info_df = pd.read_csv('plays_to_scrape.csv')
    play_name_index = play_info_df[play_info_df['title']== title].index.values
    author = play_info_df['author'][play_name_index].values[0]
    year = play_info_df['year'][play_name_index].values[0]
    genre = play_info_df['genre'][play_name_index].values[0]

    #generates csv name to look for (same rules to make the original csv name)
    play_csv_name = title
    play_csv_name = play_csv_name.replace(' ', '_')
    play_csv_name = play_csv_name.replace(',', '')
    play_csv_name = play_csv_name.replace("'", '')
    play_csv_name = play_csv_name.replace("’", '')
    play_csv_name = play_csv_name + '_df.csv'
  
    play_df = pd.read_csv(play_csv_name)
    lines = ""
    
    line_index = (play_df['text_type'] == 'Dialogue').idxmax()

    current_speaker = ""
    for i in range(3):
        is_dialogue = True
        if play_df.loc[line_index]['text_type'] == 'Dialogue':
            if play_df.loc[line_index]['speaker'] == current_speaker:
                lines = lines + "\t" +play_df.loc[line_index]['text'] + "\n"
            else:
                lines = lines + play_df.loc[line_index]['speaker'] + ": " + play_df.loc[line_index]['text'] + "\n "
                current_speaker = play_df.loc[line_index]['speaker']
            line_index = line_index + 1 
        else:
            is_dialogue = False
        
        while is_dialogue == False:
            line_index = line_index + 1
            if play_df.loc[line_index]['text_type'] == 'Dialogue':
                if play_df.loc[line_index]['speaker'] == current_speaker:
                    lines = lines + "\t" +play_df.loc[line_index]['text'] + "\n"
                else:
                    lines = lines + play_df.loc[line_index]['speaker'] + ": " + play_df.loc[line_index]['text'] + "\n "
                    current_speaker = play_df.loc[line_index]['speaker']

                is_dialogue = True
                line_index = line_index + 1
            else:
                is_dialogue = False
    
    template = get_prompt(PROMPT)
    
    prompt = template.format(title=title, year=year, author=author, genre=genre, 
                        lines_of_play=lines)
    
    next_line_index = line_index
    #careful, you'll run into an error if you run out of dialogue
    while play_df.loc[next_line_index]['text_type'] != 'Dialogue':
        next_line_index = next_line_index + 1
        
    next_line = play_df.loc[next_line_index]['text']
    
    return str(prompt), next_line

#print(create_prompt("Dido, Queen of Carthage"))