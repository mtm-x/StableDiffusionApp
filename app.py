from tkinter import *
import customtkinter 

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the app
app = customtkinter.CTk()
app.geometry("532x632")
app.title("Text to Image") 
customtkinter.set_appearance_mode("dark") 
customtkinter.set_appearance_mode("dark-blue")

prompt = customtkinter.CTkEntry(app,
    placeholder_text="Enter the prompt",
    height=40, width=480,
    #corner_radius="50",
    ) 
prompt.pack(pady=20)


modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device) 

def generate(): 
    with autocast(device): 
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img) 

def clear():
    prompt.delete(0,END)

button_s=customtkinter.CTkButton(app,text="Generate")
button_s.pack(pady=5)
button_c=customtkinter.CTkButton(app,text="Clear",command=clear)
button_c.pack(pady=5)
app.mainloop()
