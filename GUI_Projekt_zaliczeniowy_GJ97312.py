from tkinter import *
from tkinter import messagebox as msg
from PIL import Image, ImageTk

class App:

    def __init__(self, parent):

        self.widgets(parent)

    def widgets(self, app):

        self.image1 = PhotoImage(file = "Aplikacja_meme.png")

        self.photo = Label(app,
                           image = self.image1).place(x = 0,
                                                           y = 0)

        self.title = Label(app, text = 'Podaj dane do analizy procesu kredytowego').grid(row = 0,
                                                                                         column = 1)

        self.title2 = Label(app, text = 'Imię').grid(row = 1,
                                                     column = 0)
        self.name = StringVar()

        self.title3 = Label(app, text='Dochód').grid(row = 2,
                                                     column = 0)
        self.income = IntVar()

        self.title4 = Label(app,
                            text='Typ umówy').grid(row = 3,
                                                        column = 0)
        self.umowa = StringVar()

        self.title5 = Label(app,
                            text='Wiek').grid(row = 4,
                                                   column = 0)
        self.wiek = StringVar()

        self.title6 = Label(app,
                            text='Wnioskowana kwota').grid(row=5,
                                                           column=0)
        self.kwota = IntVar()

        self.entry = Entry(app,
                           textvariable = self.name).grid(row = 1,
                                                               column = 1)
        self.entry2 = Entry(app,
                            textvariable = self.income).grid(row = 2,
                                                                  column = 1)
        self.entry3 = Entry(app,
                            textvariable = self.umowa).grid(row = 3,
                                                            column = 1)
        self.entry4 = Entry(app,
                            textvariable = self.wiek).grid(row = 4,
                                                                column = 1)
        self.entry5 = Entry(app,
                            textvariable=self.kwota).grid(row=5,
                                                               column=1)
        btn = Button(app,
                     text = 'Przekaż dane',
                     command = self.decyzja_kredytowa).grid(row = 6,
                                                            column = 1)

    def decyzja_kredytowa(self):

        name = self.name.get()

        income = self.income.get()

        umowa = self.umowa.get()

        income = self.income.get()

        wiek = self.wiek.get()

        kwota = self.kwota.get()

        msg.showinfo('Informacja z systemu', 'Panie / Pani {} przekażemy dane do analityka kredytowego'.format(name.upper()))

        msg.showinfo('Informacja z systemu', 'Przekazuję info o dochodzie w wysokości {}'.format(income))

        msg.showinfo('Informacja z systemu', 'Przekazuję info o umowie typu {}'.format(umowa.upper()))

        msg.showinfo('Informacja z systemu', 'Przekazuję info o wieku {} lat'.format(wiek.upper()))

        msg.showinfo('Informacja z systemu', 'Dziękujęmy za skorzystanie z naszej bankowości'.format(wiek.upper()))

        if income/kwota < 10:
            msg.showinfo('Informacja kredytowa',
                         'Panie / Pani {} nie ma Pan / Pani zdolności kredytowej'.format(name.upper()))
        else:
            msg.showinfo('Informacja kredytowa',
                         'Kwota w wysokości {} została Panu / Pani przyznana'.format(kwota))

root = Tk()

obj = App(root)

root.mainloop()

