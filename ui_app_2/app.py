import intent
import sentiment
import stt
import query_response

from tkinter import *
from PIL import Image, ImageTk

import tkinter as tk
import pyaudio
import wave
import threading
import os
from gtts import gTTS
import pygame
import time

BG_GRAY = "#2B2A4C"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Roboto 15"
FONT_BOLD = "Roboto 15 bold"

conv_dict = {
    "intent": "",
    "sentiment": "",
    "response": ""
}

def check_file_exists(file_path):
    return os.path.exists(file_path)

def get_intent(msg):
    return intent.predict_intent(msg)

def get_sentiment(msg):
    return sentiment.predict_sentiment(msg)

def get_response(msg):
    return query_response.predict_query_response(msg)

def save_sound():
    tts = gTTS(conv_dict["response"], lang="en")
    tts.save("hello.mp3")

def play_sound():
    pygame.mixer.init()
    sound = pygame.mixer.Sound("hello.mp3")
    sound.play()
    pygame.time.wait(int(sound.get_length() * 1000))
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove("hello.mp3")
    

class AudioRecorder:
    
    def __init__(self, filename="recorded_audio.wav", format=pyaudio.paInt16, channels=1, rate=44100, frames_per_buffer=1024):
        self.filename = filename
        self.format = format
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer

        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []

        self.is_recording = False

    def start_recording(self):
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.frames_per_buffer)
        self.is_recording = True

        print("Recording...")

        while self.is_recording:
            data = self.stream.read(self.frames_per_buffer)
            self.frames.append(data)

    def stop_recording(self):
        self.is_recording = False
        print("Recording stopped.")

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()

        self.save_to_file()

    def save_to_file(self):
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))

class ChatApplication:
        
    def __init__(self):
        self.window = Tk()
        self.audio_recorder = AudioRecorder()
        image = Image.open("./mic2.png")
        self.tk_image = ImageTk.PhotoImage(image)
        self._setup_main_window()
            
    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Conversational AI")
        self.window.iconbitmap("./mic.ico")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=650, height=650, bg=BG_COLOR)
        
        self.intent_text = StringVar()
        self.sentiment_text = StringVar()

        head_label1 = Label(
            self.window,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            text="Intent: ",
            font=FONT_BOLD,
            pady=7,
        )
        head_label1["foreground"] = "orange"
        head_label1.place(relx=0.1)

        head_label2 = Label(
            self.window,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            textvariable=self.intent_text,
            font=FONT_BOLD,
            pady=7,
        )
        head_label2.place(relx=0.2)

        head_label3 = Label(
            self.window,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            text="Sentiment: ",
            font=FONT_BOLD,
            pady=7,
        )
        head_label3["foreground"] = "orange"
        head_label3.place(relx=0.5)

        head_label4 = Label(
            self.window,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            textvariable=self.sentiment_text,
            font=FONT_BOLD,
            pady=7,
        )
        head_label4.place(relx=0.66)

        self.text_widget = Text(
            self.window,
            width=20,
            height=2,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            font=FONT,
            padx=5,
            pady=5,
        )

        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        bottom_label = Label(self.window, bg=BG_GRAY, height=70)
        bottom_label.place(relwidth=1, rely=0.825)

        self.msg_entry = Entry(bottom_label, bg="#2c3e50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.64, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        send_button = Button(
            bottom_label,
            text="SEND",
            font=FONT_BOLD,
            width=150,
            bg=BG_GRAY,
            command=lambda: self._on_enter_pressed(None),
        )
        send_button["foreground"] = "white"
        send_button.place(relx=0.67, rely=0.008, relheight=0.06, relwidth=0.20)

        self.record_button = tk.Button(bottom_label, text=" Record ", font=FONT_BOLD, bg=BG_GRAY, width=200, command=self.toggle_recording)
        self.record_button.place(relx=0.88, rely=0.008, relheight=0.06, relwidth=0.11)
        self.record_button["foreground"] = "white"
        
    def toggle_recording(self):
        if not self.audio_recorder.is_recording:
            self.record_button.configure(text="❌")
            self.audio_recorder = AudioRecorder()
            self.record_thread = threading.Thread(target=self.audio_recorder.start_recording)
            self.record_thread.start()
        else:
            self.record_button.configure(text="Record")
            self.audio_recorder.stop_recording()
                
    def _on_enter_pressed(self, event):
        
        if (self.msg_entry.get() == ""):            
            if check_file_exists("recorded_audio.wav"):
                text_output = stt.predict_stt("recorded_audio.wav")
                self._insert_message(text_output, "You")
                
                conv_dict["intent"] = get_intent(text_output)
                conv_dict["sentiment"] = get_sentiment(text_output)
                print(conv_dict)
                self.intent_text.set(conv_dict["intent"])
                self.sentiment_text.set(conv_dict["sentiment"])
                os.remove("recorded_audio.wav")
        else:    
            msg = self.msg_entry.get()
            self._insert_message(msg, "You")
            conv_dict["intent"] = get_intent(msg)
            conv_dict["sentiment"] = get_sentiment(msg)
            self.intent_text.set(conv_dict["intent"])
            self.sentiment_text.set(conv_dict["sentiment"])
            
    # def _insert_message(self, msg, sender):
    #     if not msg:
    #         return
        
    #     self.msg_entry.delete(0, END)
    #     msg1 = f"{sender}: {msg}\n"
    #     self.text_widget.configure(state=NORMAL)
    #     self.text_widget.insert(END, msg1)
    #     self.text_widget.configure(state=DISABLED)
        
    #     conv_dict["response"] = get_response(msg)
        
    #     msg2 = f"Support Staff: {conv_dict['response']} \n\n"        
    #     self.text_widget.configure(state=NORMAL)
    #     self.text_widget.insert(END, msg2)
    #     self.text_widget.configure(state=DISABLED)
    #     self.text_widget.see(END)
        
    #     save_sound()
        
    #     tts_thread_play = threading.Thread(target=play_sound)
    #     tts_thread_play.start()
    
    def _insert_message(self, msg, sender):
        if not msg:
            return
    
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n"
    
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
    
        def insert_msg2():
            conv_dict["response"] = get_response(msg)
            msg2 = f"Support Staff: {conv_dict['response']} \n\n"
            self.text_widget.configure(state=NORMAL)
            self.text_widget.insert(END, msg2)
            self.text_widget.configure(state=DISABLED)
            self.text_widget.see(END)
            save_sound()
            tts_thread_play = threading.Thread(target=play_sound)
            tts_thread_play.start()

        # insert_msg2()
        # Schedule insertion of msg2 after 1000 milliseconds (1 second)
        self.text_widget.after(250, insert_msg2)

                
if __name__ == "__main__":
    app = ChatApplication()
    app.run()
    img = Image.open("./mic2.png")