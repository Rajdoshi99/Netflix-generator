# Run by typing python3 main.py

# **IMPORTANT:** only collaborators on the project where you run
# this can access this web server!

"""
    Bonus points if you want to have internship at AI Camp
    1. How can we save what user built? And if we can save them, like allow them to publish, can we load the saved results back on the home page? 
    2. Can you add a button for each generated item at the frontend to just allow that item to be added to the story that the user is building? 
    3. What other features you'd like to develop to help AI write better with a user? 
    4. How to speed up the model run? Quantize the model? Using a GPU to run the model? 
"""

# import basics
import os

# import stuff for our web server
from flask import Flask, request, redirect, url_for, render_template, session
from utils import get_base_url
# import stuff for our models
from aitextgen import aitextgen

# load up a model from memory. Note you may not need all of these options.
# ai = aitextgen(model_folder="model/",
#                tokenizer_file="model/aitextgen.tokenizer.json", to_gpu=False)

#ai = aitextgen(model="distilgpt2", to_gpu=False)

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 1234
base_url = get_base_url(port)


# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

app.secret_key = os.urandom(64)

# set up the routes and logic for the webserver




@app.route(f'{base_url}')
def home():
    return render_template('writer_home.html', generated=None)

@app.route(f'{base_url}/results')
def results():
    return render_template('Write-your-story-with-AI.html')

#@app.route(f'{base_url}')
#def my_form():
#    return render_template('index.html')

@app.route(f'{base_url}/results', methods=['POST'])
def my_form_post():
    from flask import Flask, render_template , request 
    from flask import jsonify
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import numpy as np
    import random
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
    from tqdm import tqdm, trange
    import torch.nn.functional as F
    import csv
    import os
    from sklearn.model_selection import train_test_split
    from aitextgen import aitextgen
    from aitextgen.TokenDataset import TokenDataset
    from aitextgen.utils import build_gpt2_config
    from aitextgen.tokenizers import train_tokenizer
    text=request.form['text']
    prefix=text
    '''netflix_data=pd.read_csv('netflix_titles.csv')
    file_name = "tokenizer_input.txt"                      #Can also use df.csv() to convert it into text file.
    with open(file_name, 'a') as f:
        dfAsString = netflix_data['description'].to_string(header=False, index=False)
        f.write(dfAsString)
    training_samples = netflix_data.description.values.tolist()
    training_file = TokenDataset(texts=training_samples, save_cache=False) #not using a custom tokenizer so redefining this variable to match that.
    #training_file = "dataset_cache.tar.gz"
    ai = aitextgen(tf_gpt2="124M", to_gpu=True) #Downloading the model 
    ai.train(training_file,
          line_by_line=False,
          from_cache=True,
          num_steps=3,
          generate_every=1000,
          save_every=1000,
          save_gdrive=False,
          learning_rate=1e-3,
          fp16=False,
          batch_size=1,
          )'''

    ai = aitextgen(model_folder="model/")
    b=[]
    ai.generate_to_file(n=10,
                batch_size=1,
                prompt=prefix,
              temperature=0.7,
              top_p=0.9,
              destination_path="noob.txt",
              prefix = '<|startoftext|>',
              truncate='<|endoftext|>')

    samples = open("noob.txt", 'r').read().split('====================')[:-1]
    new_text = ""

        # Format output
    for i, sample in enumerate(samples):

        new_text = f"{new_text } Sample {i+1} \n   {sample} \n "
        print("-------------------------")

    new_text = new_text.replace("\n", "")
    default_prompt = prefix
    prefix = prefix.replace("\n","</br>")
    print(new_text)

    return render_template('Write-your-story-with-AI.html', processed_text=new_text.split('Sample'))


'''
@app.route(f'{base_url}', methods=['POST'])
def home_post():
    return redirect(url_for('my_form_post'))



@app.route(f'{base_url}', methods=['POST'])
def home_post():
    return redirect(url_for('results'))


@app.route(f'{base_url}/results/')
def results():
    if 'data' in session:
        data = session['data']
        return render_template('Write-your-story-with-AI.html', generated=data)
    else:
        return render_template('Write-your-story-with-AI.html', generated=None)


@app.route(f'{base_url}/generate_text/', methods=["POST"])
def generate_text():
    """
    view function that will return json response for generated text. 
    """

    prompt = request.form['prompt']
    if prompt is not None:
        generated = ai.generate(
            n=1,
            batch_size=3,
            prompt=str(prompt),
            max_length=300,
            temperature=0.9,
            return_as_list=True
        )

    data = {'generated_ls': generated}
    session['data'] = generated[0]
    return redirect(url_for('results'))'''


# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page


if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalcg1.ai-camp.dev'

    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host='0.0.0.0', port=port, debug=True)
