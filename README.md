**NETFLIX TEXT-GENERATOR**!

Given a keyword or a prompt, a pre-trained GPT-2 model is fine-tuned to generate brand new movie/tv descriptions. The model has been trained to output the next word/text -it knows exactly where to add spaces because it has learnt these patterns of spaces between chunks of characters– it also knows when to add periods because it has noticed periods at the end of sentences– and similarly is also smart enough to capitalize the first letter of each sentence. This model will aid in the production of some fascinating stories and Netflix descriptions for new series and movies based on previous or present shows and movies, and so on. It can also be used to generate fan theories revolving around different movies, their prequels , sequels and plots for open ended movies. 

**Data:**

The model is trained on the Netflix dataset obtained from Kaggle. The dataset contains approximately 8000 movies and television series that are available on their site. This tabular dataset contains lists of all the movies and TV series accessible on Netflix, together with information such as actors, directors, ratings, release year, duration,etc.
Link- https://www.kaggle.com/datasets/shivamb/netflix-shows

**Advanced Features:**

The Advanced features of the model could be based on the director or year, our model could generate descriptions of various shows or movies, etc. For example , prompt(
“Michael Filangan in 2018”) or prompt(“Love in 2017”) it would generate descriptions of plots or new theories based on the open-ended movies, shows, etc by taking into account the years and the genre and provide us with concrete text description. There’s even the ability to extend this further by having text to voice technology.

**Evaluation Metric:**

The BLEU (BiLingual Evaluation Understudy) score will be used for evaluation of the model. This metric automatically evaluates machine-translated text by giving it a score between 0 and 1  based on the similarity of the machine-translated text to a set of high quality reference translations.

**References:**

1.https://towardsdatascience.com/how-to-fine-tune-gpt-2-for-text-generation-ae2ea53bc272
2.https://colab.research.google.com/drive/1yVxt4TwdZPQ4e1F7I9EG52rSzyu7EAJ_#sandboxMode=true&scrollTo=Ooubrodvm6QD

**OUTPUT:-**

![noob](https://user-images.githubusercontent.com/44093439/164945343-a63d4328-235d-419d-9e4e-041eb8b776c5.jpg)
