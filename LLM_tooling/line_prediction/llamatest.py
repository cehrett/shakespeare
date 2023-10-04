import llm_prompt
from csv import writer

# Change play that is predicted here
title = "Dido, Queen of Carthage"
num_of_responses = 5

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

token = "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"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token = token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token = token, cache_dir="/scratch1/sdj5/hf")

# creates the prompt and next line and stores it as a string
new_prompt, next_line = llm_prompt.create_prompt(title)
new_prompt = str(new_prompt)

# to store the prompt (for some reason it gives you nonsense if you don't do it this way)
p = new_prompt

inputs = tokenizer(str(new_prompt), return_tensors="pt")

# generates the given number of responses and stores it in a csv
for i in range (num_of_responses):
    generate_ids = model.generate(inputs.input_ids)
    output = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

    # gets the output as a string, and stores the title, prompt, model, output, and true next line in a csv
    o = str(output).replace(p.strip(), '')
    row = [title, p, 'Llama 2', o, next_line]

    # open the file in the write mode
    with open('output.csv', 'a', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer_object  = writer(f)

        # write a row to the csv file
        writer_object.writerow(row)

        f.close()