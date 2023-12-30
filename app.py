import gradio as gr
import numpy as np
from transformers import pipeline

title = "Translation from English to French"
description = """
Example-> 

Hello, My name is Rahul => Bonjour, je m'appelle Rahul

<img src="https://huggingface.co/spaces/course-demos/Rick_and_Morty_QA/resolve/main/rick.png" width=200px>
"""

article = "Check out [my github repository](https://github.com/Neural-Net-Rahul/P4-Translation-using-fine-tuned-hugging-face-transformer) and my [fine tuned model](https://huggingface.co/neural-net-rahul/marian-finetuned-kde4-en-to-fr)"

textbox = gr.Textbox(label="Type your sentence here :", placeholder="My name is Bill Gates.", lines=3)

model = pipeline('translation',model='neural-net-rahul/marian-finetuned-kde4-en-to-fr')

def tellFrench(text):
  return model(text)[0]['translation_text']


gr.Interface(
    fn=tellFrench,
    inputs=textbox,
    outputs="text",
    title=title,
    description=description,
    article=article,
    examples=[["Musk took an active role within the company and oversaw Roadster product design,but was not deeply involved in day-to-day business operations"], ["A 2009 lawsuit settlement with Eberhard designated Musk as a Tesla co-founder, along with Tarpenning and two others."]],
).launch(share=True)
