################## Interfaz
import tkinter as tk
from tkinter import ttk


class SIS_interfaz(tk.Tk):
    
    def __init__(self):
        tk.Tk.__init__(self)
        self.wm_title("Coopetitivo")
        return(self.combo())
        
    def combo(self):
        
        self.containter = tk.Frame(self, height=300, width=600, bg="azure")
        self.containter.pack(fill="both", expand=True)
        
        self.combo_01 = ttk.Combobox(self.containter,
                                  values=["Competitivo", "Cooperativo"])
        self.combo_01.set("Competitivo")
        self.combo_01.pack(pady = 10, padx = 10)
        
        self.combo_02 = ttk.Combobox(self.containter,
                                     values=["Vertical","Horizontal"])
        self.combo_02.set("Vertical")
        self.combo_02.pack(pady = 10, padx = 10)
        
        self.boton_01 = tk.Button(self.containter,
                                  text="Siguiente",
                                  command=lambda : [self.combo1(),
                                                    self.containter.destroy()],
                                  bg="azure")
        self.boton_01.pack()
        
        
    def combo1(self):
        
        self.combo_02 = self.combo_02.get()
        self.containter1 = tk.Frame(self, height=300, width=600, bg="azure")
        
        vals = list(caracteristica.values())[2:6]#Capacidades
        
        self.combo_11 = ttk.Combobox(self.containter1,
                                     values=vals)
        self.combo_11.set(vals[0])
        self.combo_11.pack(padx=10, pady=10)
        
        vals2 = np.unique(list(M_caracteristicas.iloc[:, 1][M_caracteristicas.iloc[:,1]!="Proveedor "]))
        vals2 = list(vals2)
        self.combo_12 = ttk.Combobox(self.containter1,
                                     values=vals2)#Tipo de actor
        
        if self.combo_02 == "Horizontal":
        
            self.combo_12.set(vals2[0])
            self.combo_12.pack(padx=10, pady=10)
        
        self.combo_13 = ttk.Combobox(self.containter1,
                                     values=["Manual", "Pseudoaleatorio"])#Seleccion
        self.combo_13.set(self.combo_13["values"][1])
        self.combo_13.pack(padx=10, pady=10)
        
        
        self.containter1.pack(fill="both", expand=True)
        
        
        self.boton_11 = tk.Button(self.containter1, text = "Siguiente/Ok",
                                  command=lambda :[self.pasar(),
                                                   self.containter1.destroy()
                                                   ],
                                  bg="azure")
        
        self.boton_11.pack()
        
    def pasar(self):
        
        self.combo_12 = self.combo_12.get()
        
        self.ent = Sc.get_rows(M_caracteristicas, self.combo_12)
        
        if self.combo_12 == "":
            
            self.combo_12 = "Grafo vertical I+D"
        
        if self.combo_13.get() == "Manual":
            
            return(self.combo2())
        
        else:
            
            return(self.graph())
        
    def combo2(self):
        
        self.containter2 = tk.Frame(self, bg = "azure")
        
        matriz = M_conex.loc[self.ent,self.ent]
        
        vals = [i[0] for i in matriz.index]
        
        self.combo_21 = ttk.Combobox(self.containter2, values=vals)
        
        self.combo_21.set(self.combo_21["values"][0])
        
        self.combo_21.pack(pady=10, padx=10)
        
        self.boton21 = tk.Button(self.containter2, bg = "azure",
                                 text = "Ok",
                                 command = lambda: [self.graph(),
                                                    self.containter2.destroy()])
        
        self.boton21.pack(pady=10, padx=10)
        
        self.containter2.pack(fill="both", expand=True)
        
    def graph(self):
        
        self.fig_1 = mtp.figure.Figure()
        
        self.sis = self.fig_1.add_subplot(111)
        
        Sc.graph(M_conex.loc[self.ent,self.ent], M_caracteristicas.iloc[:,1])
        plt.savefig("out/Graph_3.png", bbox_inches="tight", dpi=1200)
        plt.close()
        
        img_arr = mtp.image.imread('out/Graph_3.png')
        self.sis.imshow(img_arr)
        self.sis.set_title(self.combo_12)
        
        self.sis.spines["top"].set_visible(False)
        self.sis.spines["left"].set_visible(False)
        self.sis.spines["right"].set_visible(False)
        self.sis.spines["bottom"].set_visible(False)
        self.sis.xaxis.set_visible(False)
        self.sis.yaxis.set_visible(False)
        
        canvas = FigureCanvasTkAgg(self.fig_1, master=self)
        toolbar = NavigationToolbar2Tk(canvas, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        canvas._tkcanvas.pack(side="left", fill="both", expand=True)
        
        #np.linalg.eig(
        #Sc.mat_power(5 ,
        #Sc.diferencias(
        #Sc.make_jaccard(M_conex, 
        #coeficientes.loc[:,"Capacidad de investigacion"]), 
        #coeficientes.loc[:,"Capacidad de investigacion"], 
        #list(M_conex.sum()))[0]))
        
        
    
        
        
        
        
        
if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
    import matplotlib as mtp
    import copy
    import Simulacion_coopetitivo as Sc
    
##################Globalizacion solo para agilizar

    global caracteristica
    global M_caracteristicas
    global M_conex
    global coeficientes
    global pesos
    global pef

     
##################Diccionario de datos

    caracteristica = {"1":"Nro de topicos","2":"Tipo de actor", "3":"Capacidad de investigacion",
"4":"Capacidad de desarrollo", "5":"Capacidad de difusion", "6":"Capacidad de mercadeo"}

#################

################# Matriz de caracteristicas y conexiones

    M_caracteristicas = pd.read_csv("in/Matriz de CTI-ok.csv",sep=",",
                                header=None, index_col=0, decimal=",")
    M_caracteristicas.columns=[list(caracteristica.values())]

    entidades = {f"{i+1}":
             M_caracteristicas.index[i] for i in range(len(M_caracteristicas.index))}

    M_caracteristicas.reset_index(drop=True, inplace=True)

    M_caracteristicas.index=[list(entidades.values())]

    M_conex = pd.read_csv("in/matrizactores.csv", sep=",", header=None)

    M_conex.columns=[list(entidades.values())]

    M_conex.index=[list(entidades.values())]

    M_conex = Sc.corrector(M_conex)

################## Coeficientes

    coeficientes = M_caracteristicas[M_caracteristicas.columns[-4:]]

################# Pesos

    pesos = []        
    pef = []

    for i in M_conex.columns:

        pf = M_conex.transpose()[i].sum()

        pef.append(pf)

        pesos.append(2*pf)
    
    root = SIS_interfaz()
    root.mainloop()
    