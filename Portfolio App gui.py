from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
import _thread

import financial_agent
import data_agent
import LSTM_predictor
from tkinter import messagebox

import trade_manager

root = Tk()
root.title("Portfolio App")
root.iconbitmap('C:\PythonPrograms\Bakalauro Baigiamasis Darbas\LSTM stock price prediction\money.ico')
root.geometry("600x600")

my_menu=Menu(root)
root.config(menu=my_menu)

def our_command():
    print("Hello Portfolio App!")

profile_menu= Menu(my_menu)
my_menu.add_cascade(label="Apžvalga", menu=profile_menu, font="-weight bold")
profile_menu.add_command(label="New...", command=our_command)
profile_menu.add_separator()
profile_menu.add_command(label="Exit", command=root.quit)

edit_menu = Menu(my_menu)
my_menu.add_cascade(label="Bankininkystė",menu=edit_menu)
edit_menu.add_command(label="Cut",command=our_command)
edit_menu.add_command(label="Copy",command=our_command)

option_menu = Menu(my_menu)
my_menu.add_cascade(label="Investavimas",menu=option_menu)
option_menu.add_command(label="Find",command=our_command)
option_menu.add_command(label="Find Next",command=our_command)

option_menu = Menu(my_menu)
my_menu.add_cascade(label="Planavimas",menu=option_menu)
option_menu.add_command(label="Find",command=our_command)
option_menu.add_command(label="Find Next",command=our_command)

option_menu = Menu(my_menu)
my_menu.add_cascade(label="Patarimai",menu=option_menu)
option_menu.add_command(label="Find",command=our_command)
option_menu.add_command(label="Find Next",command=our_command)



# Grynoji verte (samata) frame
def get_funds():
    return (financial_agent.get_buying_power() + " $")
samata_frame = LabelFrame(root, text="Grynoji vertė", font=("-weight bold", 20), pady=20)
suma_label = Label(samata_frame, text=get_funds(), font=("-weight bold", 15)).grid(row=0, column=0)

samata_frame.grid(row=0, column=0, padx=10, pady=10)

companies_frame = LabelFrame(root, pady=20)
# configure the grid layout
companies_frame.rowconfigure(0, weight=1)
companies_frame.columnconfigure(0, weight=1)

# create a treeview
tree = ttk.Treeview(root)
tree.heading('#0', text='Investicijų valdymas', anchor='w')

# adding data
tree.insert('', END, text='Grynieji', iid=0, open=False)
tree.insert('', END, text='Investuoti', iid=1, open=False)
tree.insert('', END, text='Kreditas', iid=2, open=False)

# adding children of first node
tree.insert('', END, text='GOOGL', iid=5, open=False)
tree.insert('', END, text='MSFT', iid=6, open=False)
tree.insert('', END, text='AAPL', iid=7, open=False)
tree.insert('', END, text='TSLA', iid=8, open=False)
tree.move(5, 1, 0)
tree.move(6, 1, 1)
tree.move(7, 1, 2)
tree.move(8, 1, 3)


# place the Treeview widget on the root window
tree.grid(row=1, column=0, sticky='nsew')


# Profile name frame
profile_frame = LabelFrame(root)
image = Image.open("profile pic.ico")
photo = ImageTk.PhotoImage(image.resize((45, 45), Image.ANTIALIAS))
label = Label(profile_frame, image=photo)
label.image = photo
label.grid(row=0, column=0, padx=10, pady=10)
first_label1 = Label(profile_frame, text="Administratorius")
first_label1.grid(row=0, column=1, padx=(0, 10))
profile_frame.grid(sticky="e", row=0, column=2, padx=10, pady=10)


# Company name big label

def setCompanyName(newName):
    company_name_label = Label(root, text=newName, font=("-weight bold", 20))
    company_name_label.grid(row=0, column=1)
setCompanyName("                         ")

def display_close_price_graph(company):
    if company == "": return
    else:
        data_agent.get_stock_data(company)
        data_agent.close_price_history(company)

def get_selected_company():
    selection = tree.focus()
    return tree.item(selection).get('text')

def selectItem():
    company = get_selected_company()
    # print(tree.item(selection))
    setCompanyName("              ")
    setCompanyName(company)
    _thread.start_new_thread(display_close_price_graph, (company, ))

def select_rule(company, strategy):
    messagebox.showinfo("Informacija", "Pasirinkote " +strategy+ " investavimo strategiją " +company+ " įmonei.")
    rule_dialog.destroy()
    if strategy=="LSTM Rekurentinis Neuroninis Tinklas":
        _thread.start_new_thread(LSTM_predictor.calculate_LSTM, (company, ))


def open_new_rule_dialog():
    # Toplevel object which will
    # be treated as a new window
    global rule_dialog
    rule_dialog = Toplevel(root)

    # sets the title of the
    # Toplevel widget
    rule_dialog.title("Portfolio App - Sukurti naują taisyklę")

    rule_dialog.iconbitmap('C:\PythonPrograms\Bakalauro Baigiamasis Darbas\LSTM stock price prediction\money.ico')
    # sets the geometry of toplevel
    rule_dialog.geometry("450x400")

    # A Label widget to show in toplevel
    Label(rule_dialog, text="Pasirinkta įmonė: ").grid(sticky="w", row=0, column=0, padx=10, pady=10)
    global company_name
    company_name = Entry(rule_dialog, width=30, borderwidth=2)
    company_name.insert(0, get_selected_company())
    company_name.grid(sticky="w", row=0, column=1, padx=(0, 10), pady=10)
    Label(rule_dialog, text="Pasirinkta investavimo strategija: ").grid(sticky="w", row=1, column=0, padx=10, pady=10)
    options = [
        "LSTM Rekurentinis Neuroninis Tinklas",
        "Fibonacci Retracement Lygiai",
        "Vėžlių Prekyba",
        "Ilgalaikė investicija"
    ]

    global clicked
    clicked = StringVar()
    clicked.set(options[0])

    drop = OptionMenu(rule_dialog, clicked, *options)
    drop.grid(sticky="w", row=1, column=1, padx=(0, 10), pady=10)

    chooseStrategyButton = Button(rule_dialog, text="Įtraukti taisyklę", command=lambda: select_rule(company_name.get(), clicked.get()))
    chooseStrategyButton.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

def ask_directory():
    return filedialog.askdirectory()


def test_and_predict(model, company):
    LSTM_predictor.test_model(model, company)
    prediction_for_tomorrow = LSTM_predictor.predictOneDay(model, company)
    print(prediction_for_tomorrow)
    real_value_for_today = LSTM_predictor.getRealValue(company)
    print(real_value_for_today)
    trade_manager.execute_LSTM_strategy(company, float(real_value_for_today), float(prediction_for_tomorrow))

def test_model():
    path = ask_directory()
    model = data_agent.load_saved_agent(path)
    company = get_selected_company()
    _thread.start_new_thread(test_and_predict, (model, company, ))

selecttreeitembutton = Button(root, text="Parodyti uždarymo kainų grafiką", command=selectItem)
selecttreeitembutton.grid(row=2, column=0, padx=10, pady=10)
createRuleButton = Button(root, text="Sukurti naują taisyklę", command=open_new_rule_dialog)
createRuleButton.grid(row=3, column=0, padx=10, pady=(0, 10))
selectModelButton = Button(root, text="Pritaikyti LSTM modelį", command=test_model)
selectModelButton.grid(row=4, column=0, padx=10, pady=(0, 10))

root.mainloop()
