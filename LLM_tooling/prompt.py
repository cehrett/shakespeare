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


def create_prompt(title, year, author, genre):
    import pandas as pd

    play_df = pd.read_csv(r'C:\Users\sammy\OneDrive\Desktop\Clemson Stuff\INNO 3990 (CI)\Shakespeare\Code\Dido_Queen_of_Carthage_df.csv')
    lines = ""

    line_index = (play_df['text_type'] == 'Dialogue').idxmax()
    for i in range(3):
        is_dialogue = True
        if play_df.loc[line_index]['text_type'] == 'Dialogue':
            lines = lines + play_df.loc[line_index]['text'] + " / "
            line_index = line_index + 1
        else:
            is_dialogue = False
        
        while is_dialogue == False:
            line_index = line_index + 1
            if play_df.loc[line_index]['text_type'] == 'Dialogue':
                lines = lines + play_df.loc[line_index]['text'] + " / "
                is_dialogue = True
                line_index = line_index + 1
            else:
                is_dialogue = False
    
    template = get_prompt(PROMPT)
    #print(template)

    # prompt = PROMPT.format(title=title, year=year, author=author, genre=genre, 
    #                    lines_of_play=lines)
    prompt = template.format(title=title, year=year, author=author, genre=genre, 
                        lines_of_play=lines)
    # prompt = template.format
    # prompt = PromptTemplate(template=template, input_variables=[title, year, author, genre, lines])
    
    #print(prompt)
    return prompt

print(create_prompt("Dido, Queen of Carthage", "1585-1586", "Christopher Marlowe","Tragedy"))