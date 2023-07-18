from tkinter import *
from PIL import Image, ImageTk

BG_GRAY = "#2B2A4C"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Roboto 14"
FONT_BOLD = "Roboto 14 bold"


class ChatApplication:
    def __init__(self):
        self.window = Tk()
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

        head_label = Label(
            self.window,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            text="Intent: ",
            font=FONT_BOLD,
            pady=7,
        )
        head_label["foreground"] = "orange"
        head_label.place(relx=0.1)

        head_label = Label(
            self.window,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            text="",
            font=FONT_BOLD,
            pady=7,
        )
        head_label.place(relx=0.2)

        head_label = Label(
            self.window,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            text="Sentiment: ",
            font=FONT_BOLD,
            pady=7,
        )
        head_label["foreground"] = "orange"
        head_label.place(relx=0.5)

        head_label = Label(
            self.window,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            text="",
            font=FONT_BOLD,
            pady=7,
        )
        head_label.place(relx=0.66)

        # line = Label(self.window, width=450, bg=BG_GRAY)
        # line.place(relwidth=1, rely=0.07, relheight=0.012)

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
            width=20,
            bg=BG_GRAY,
            command=lambda: self._on_enter_pressed(None),
        )
        send_button["foreground"] = "white"
        send_button.place(relx=0.67, rely=0.008, relheight=0.06, relwidth=0.20)

        send_button = Button(
            bottom_label,
            text="VOICE",
            image=self.tk_image,
            font=FONT_BOLD,
            width=20,
            bg=BG_GRAY,
            command=lambda: self._on_enter_pressed(None),
        )
        send_button["foreground"] = "white"
        send_button.place(relx=0.88, rely=0.008, relheight=0.06, relwidth=0.11)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")

    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        msg2 = f"Hari: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)


if __name__ == "__main__":
    app = ChatApplication()
    app.run()
    img = Image.open("./mic.png")
    tk_image = ImageTk.PhotoImage(img)
