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
                                  values=["Competitivo", "Cooperativo"],
                                   state="readonly")
        self.combo_01.set("Competitivo")
        self.combo_01.pack(pady = 10, padx = 10)
        
        self.combo_02 = ttk.Combobox(self.containter,
                                     values=["Vertical","Horizontal"],
                                      state="readonly")
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
                                     values=vals,  state="readonly")
        self.combo_11.set(vals[0])
        self.combo_11.pack(padx=10, pady=10)
        
        vals2 = np.unique(list(M_caracteristicas.iloc[:, 1]))
        vals2 = list(vals2)
        self.combo_12 = ttk.Combobox(self.containter1,
                                     values=vals2,  state="readonly")#Tipo de actor
        
        if self.combo_02 == "Horizontal":
        
            self.combo_12.set(vals2[0])
            self.combo_12.pack(padx=10, pady=10)
            
        elif self.combo_02 == "Vertical":
            
            self.combo_121 = ttk.Combobox(self.containter1,
                                          values = vals2, state="readonly")
            self.combo_121.set(vals2[0])
            self.combo_121.pack(padx=10, pady=10)
            
            self.combo_122 = ttk.Combobox(self.containter1,
                                          values = vals2, state="readonly")
            self.combo_122.set(vals2[1])
            self.combo_122.pack(padx=10, pady=10)
            
            self.combo_123 = ttk.Combobox(self.containter1,
                                          values = vals2, state="readonly")
            self.combo_123.set(vals2[2])
            self.combo_123.pack(padx=10, pady=10)
            
            self.combo_124 = ttk.Combobox(self.containter1,
                                          values = vals2, state="readonly")
            self.combo_124.set(vals2[3])
            self.combo_124.pack(padx=10, pady=10)
            
            self.combo_125 = ttk.Combobox(self.containter1,
                                          values = vals2, state="readonly")
            self.combo_125.set(vals2[4])
            self.combo_125.pack(padx=10, pady=10)
            
        
        self.combo_13 = ttk.Combobox(self.containter1,
                                     values=["Manual", "Pseudoaleatorio"],
                                      state="readonly")#Seleccion
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
        
        if self.combo_02=="Horizontal":
            
            self.combo_12 = self.combo_12.get() #El tipo de entidad
            
        elif self.combo_02=="Vertical":
            
            self.combo_12 = [self.combo_121.get(), self.combo_122.get(),
                             self.combo_123.get(), self.combo_124.get(),
                             self.combo_125.get()]
            print(self.combo_12, len(self.combo_12))
        
        self.combo_11 = self.combo_11.get() # Capacidades de interez
        
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
        
        self.combo_21 = ttk.Combobox(self.containter2, values=vals,
                                     state="readonly")
        
        self.combo_21.set(self.combo_21["values"][0])
        
        self.combo_21.pack(pady=10, padx=10)
        
        self.boton21 = tk.Button(self.containter2, bg = "azure",
                                 text = "Ok",
                                 command = lambda: [self.graph(),
                                                    self.containter2.destroy()])
        
        self.boton21.pack(pady=10, padx=10)
        
        self.containter2.pack(fill="both", expand=True)
        
    def graph(self):
        
        try: 
            self.combo_21 = self.combo_21.get()
        except:
            matriz = M_conex.loc[self.ent,self.ent]
            vals = [i[0] for i in matriz.index]
            self.combo_21 = np.random.choice(vals, 1)
            self.combo_21 = self.combo_21[0]
        
        self.fig_1 = mtp.figure.Figure()
        self.fig_2 = mtp.figure.Figure()
        
        self.sis = self.fig_1.add_subplot(111)
        self.sis_2 = self.fig_2.add_subplot(111)
        
        c_tiempo = 0.5
        duracion = 1000 # días
        poblacion = len(M_conex.loc[self.ent, self.ent])
        matriz = M_conex.loc[self.ent, self.ent]
        print(matriz)
        cof = M_caracteristicas.loc[self.ent, self.combo_11]
        Matriz = Sc.diferencias(Sc.make_jaccard(matriz, cof),
                                cof, matriz.sum())
        print(Matriz[0])
        mat_p = Sc.make_prob(Matriz[0])
        print(mat_p)
        beta = Matriz[1][self.combo_21]
        gamma = Matriz[2][self.combo_21]*beta
        
        Sc.graph(round(mat_p,5), M_caracteristicas.iloc[:,1])
        plt.savefig("out/"+f"{self.combo_12}"+".png", bbox_inches="tight", dpi=1200)
        plt.close()
        
        M_conex.loc[self.ent, self.ent].to_csv("out/"+f"{self.combo_12}"+".csv",
                                               sep=";")
        
        img_arr = mtp.image.imread("out/"+f"{self.combo_12}"+".png")
        self.sis.imshow(img_arr)
        self.sis.set_title(f"{np.unique(self.combo_12)}")
        
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
        
        
        self.sis_2.set_title(self.combo_11+" "+self.combo_21)
        self.sis_2.set_xlabel("Tiempo días")
        self.sis_2.set_ylabel("Frecuencia de infección")
        
        
        
        self.sis_2.plot(Sc.DTMC_SIS(beta, gamma, poblacion, c_tiempo, duracion)[3],Sc.DTMC_SIS(beta, gamma, poblacion, c_tiempo, duracion)[0])
        self.sis_2.plot(Sc.DTMC_SIS(beta, gamma, poblacion, c_tiempo, duracion)[3],Sc.DTMC_SIS(beta, gamma, poblacion, c_tiempo, duracion)[1])
        canvas_2 = FigureCanvasTkAgg(self.fig_2, master=self)
        toolbar_2 = NavigationToolbar2Tk(canvas_2, self)
        canvas_2.draw()
        canvas_2.get_tk_widget().pack(side="right", fill="both", expand=True)
        canvas_2._tkcanvas.pack(side="right", fill="both", expand=True)
        
        g = Sc.Graph(len(mat_p))
        g.graph=np.array(mat_p)
        dijk= g.dijkstra(1)
        print(dijk.sort_values(by="Distance from Source"))
        
        
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
    
    
    
