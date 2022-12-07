import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

def corrector(matriz):

    Matriz = copy.deepcopy(matriz)
    i = 0
    for rows in matriz.index:
        
        a = Matriz.loc[rows, :].to_numpy()[i:len(matriz.index)]
        Matriz.iloc[[v for v in range(i,len(matriz.index))], i] = a
        Matriz.loc[rows, rows] = 0
        i += 1
        
    return(Matriz)

def mat_power(n, matriz):

    m_p = copy.deepcopy(matriz)

    for i in range(n):

        m_p = m_p.dot(matriz)

    return(m_p)

def make_prob(Matriz):

    matriz = copy.deepcopy(Matriz)

    for rows in Matriz.index:

        matriz.loc[rows, :] = Matriz.loc[rows,:]/(Matriz.loc[rows,:].sum())
    
    return(matriz)

def make_jaccard(Matriz, coeficientes):

    matriz = copy.deepcopy(Matriz)

    for rows in Matriz.index:

        a = coeficientes.transpose()*Matriz.loc[rows, :]
        #print(a)
        #a.index = rows

        a = a.squeeze()

        matriz.loc[rows, :] = a.transpose()

    return (matriz)

def graph(conexiones,clase,coeficientes=[]):
    
    fig, ax = plt.subplots()
    
    if coeficientes == []:
        
        pesot = conexiones.sum()
        
    else: 
        
        pesot = conexiones.sum()*coeficientes

    nexos = conexiones.to_dict()

    pos = {}
    
    tipo = list(np.unique(clase))
    
    clase = clase
    
    entidades = {}
    
    for i in range(len(nexos)):
        
        entidades.update({f"{i+1}":conexiones.index[i][0]})
    
    
    color = {}
    
    colortipo = []
    
    colory = ["tomato", "mediumseagreen", "deepskyblue", "coral", "teal"]
    
    for i in nexos:
        
        a = colory[tipo.index(clase.loc[i,:].to_numpy()[0])]
        color.update({f"{i[0]}":
                      a})     

    c = 0
    
    u = np.random.uniform(0,1, len(nexos))

    for i in nexos:

        pos.update({f"{i[0]}" : [2*u[c], u[-c]+1]})

        c +=1
    
    edge = {}

    for i in nexos:

        edge.update({f"{i[0]}" : []})

        for j in nexos[i]:

            if nexos[i][j] != 0:

                edge[f"{i[0]}"].append(f"{j[0]}")

    #c = 0
    #k = 10*np.pi/np.max(pesot)

    for i in edge:

 #       color = plt.cm.RdYlBu(np.sin(k*int(pesot[c])))

        for j in edge[i]:
            
            valor_medio=[(pos[i][0]+pos[j][0])/2, (pos[i][1]+pos[j][1])/2]

            plt.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=color[f"{i}"], linewidth=0.2, zorder=-1)
            
            plt.text(valor_medio[0], valor_medio[1], str(conexiones.loc[i,j].to_numpy()[0][0]), fontsize=3)
       # c += 1

    c = 1

    for i in nexos:

      #  color = plt.cm.RdYlBu(np.sin(k*int(pesot[c-1])))

        size = pesot[c-1]*(len(conexiones)/43)

        plt.scatter(pos[f"{i[0]}"][0], pos[f"{i[0]}"][1], s=size, color=color[f"{i[0]}"], zorder=1)

        plt.text(pos[f"{i[0]}"][0], pos[f"{i[0]}"][1], entidades[f"{c}"], fontsize=4)

        c += 1
    
    for i in tipo:
        
        colortipo.append(plt.cm.RdYlBu(tipo.index(i)/10))
        
        #plt.scatter( 10, -2, color = colortipo[tipo.index(i)], label = i)
    
    #plt.legend(tipo, loc="lower right")
    
    
    
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    return([pos, edge, color])


def markov(init, matriz, t, condicion=0 ,estados=[]):

    if len(estados) == 0:

        estados = np.arange(len(init))

    if condicion == 0:

        condicion = estados[-1]

    estados = estados
    simlist = np.zeros(t+1)
    simlist = [list(np.random.choice(estados, 1, p=init))]

    for i in range(1, t+1):#https://people.carleton.edu/~rdobrow/StochasticBook/Rscripts/utilities.R

        if condicion in simlist:

            break

        simlist.append(list(np.random.choice(estados, 1, p=matriz.loc[simlist[i-1], :].to_numpy()[0])))

    return(simlist)

def DTMC_SIS(beta, gamma, poblacion, c_tiempo, duracion, NA=1, NB=1, JA=1, JB=1): #Markov_Chain_epidemic_models_and_Parameter_Estimation

    #Los infectados puede ser un vector donde se almacene un 
    #vector de valores unicos
    #Las reeinfecciones no se cuentan
    #Entonces podemos usar el mismo modelo
    #donde I sea el vector de infectados
    #S el vector de entidades susceptibles
    #El ultimo contagia, podrías cambiar el beta y gamma posiblemente
    I = [1]
    S = [poblacion-I[0]]
    dt = int(duracion/c_tiempo)
    M_T = np.matrix(np.zeros((dt,dt)))
    M_T[0,0]=1
    Tc = []

    tc = c_tiempo/(10)
    
    Tc.append(c_tiempo)
    #Añadir un ciclo for aqui para la lista de simulacion

    for t in range(dt):

        p1=(beta*S[t]*I[t]/poblacion)*tc

        p2=(gamma*I[t]*tc)

        q = 1-(p1+p2)

        if len(M_T)-1 >= t+1:

            M_T[t+1,t+1]=q
            M_T[t,t+1]=p2

            if len(M_T)-1 >= t+2:
                
                M_T[t+2,t+1]=p1

        #print(M_T)

        u = np.random.uniform(0, 1, 1)

        if 0 < u <= p1:

            S.append(S[t]-1)

            I.append(I[t]+1)

        elif p1 < u <= (p1+p2):

            S.append(S[t]+1)
            I.append(I[t]-1)

        else:

            S.append(S[t])
            I.append(I[t])
            
        Tc.append(Tc[t]+c_tiempo)

    return ([I,S,pd.DataFrame(M_T).transpose(),Tc])

def int_M_C(array):

    M_C = np.mean(array, axis=0)

    return(M_C)

def ver(array,el):

    if not el in array:

        array.append(el)

    return(array)

def union(arr1,arr2):
    
    arr = arr1*arr2
    
    return(arr)

def make_gameboard(Tamanio, JA):

    matriz = np.zeros((Tamanio,Tamanio))
    estados = ["S", "I"]
    estados2 = [f"a{i}" for i in range(Tamanio-len(estados))]
    estados3 = estados+estados2
    Matriz = pd.DataFrame(matriz)
    Matriz.loc[0,1] = 1
    for i in range(1,Tamanio-1):
        

        Matriz.loc[i, 0] = 1-JA
    
    for j in range(1, Tamanio-1):

        Matriz.loc[j, j+1] = JA

    Matriz.loc[j+1, 0] = 1

    Matriz.index = estados3
    Matriz.columns = estados3

    return(Matriz)

def get_rows(M_caracteristicas,cadena=""):
    
    if cadena == "":
        
        rows = [i[0] for i in 
            M_caracteristicas[M_caracteristicas.iloc[:, 1]!=0].index]
    
    elif type(cadena) == type([]):
        
        if len(cadena) == 2:
            
            rows = [i[0] for i in 
            M_caracteristicas[(M_caracteristicas.iloc[:, 1]==cadena[0]) | (M_caracteristicas.iloc[:, 1]==cadena[1])].index]
        
        elif len(cadena) == 3:
            
            rows = [i[0] for i in 
            M_caracteristicas[(M_caracteristicas.iloc[:, 1]==cadena[0]) | (M_caracteristicas.iloc[:, 1]==cadena[1]) | (M_caracteristicas.iloc[:, 1]==cadena[2])].index]
            
        elif len(cadena) == 4:
            
            rows = [i[0] for i in 
            M_caracteristicas[(M_caracteristicas.iloc[:, 1]==cadena[0])
                              | (M_caracteristicas.iloc[:, 1]==cadena[1]) 
                              | (M_caracteristicas.iloc[:, 1]==cadena[2])
                              | (M_caracteristicas.iloc[:, 1]==cadena[3])].index]
            
        elif len(cadena) == 5:
            
            rows = [i[0] for i in 
            M_caracteristicas[(M_caracteristicas.iloc[:, 1]==cadena[0])
                              | (M_caracteristicas.iloc[:, 1]==cadena[1]) 
                              | (M_caracteristicas.iloc[:, 1]==cadena[2])
                              | (M_caracteristicas.iloc[:, 1]==cadena[3])
                              | (M_caracteristicas.iloc[:, 1]==cadena[4])].index]
                                 
    #Ver como funciona para distintos or
    else:
        
        rows = [i[0] for i in 
            M_caracteristicas[M_caracteristicas.iloc[:, 1]==cadena].index]
    
    return(rows)


#def regladetres(dift, dif, prob):

 #   prox = prob*dif/dift

  #  return(prox)

def diferencias(Matriz, coeficientes, pesos_fila):
    
    #Evalue si los coeficientes de cada fila son superiores o inferiores al coeficiente de los otros
    #Si es inferior guardar en inf la resta del coeficiente i y su inferior
    #Si es superior guardar en sup la resta del superior y su coeficiente i
    #Divida la longitud del vector inf sobre el peso_fila i y guardelo en prob_E
    #Divida la longitud del vector sup sobre el peso_fila i y guardelo en prob_F

    Matriz2 = copy.deepcopy(Matriz) #Copie la matriz

    inf = {} #Cree un diccionario de nombre inf
    sup = {} #Cree un diccionario de nombre sup
    prob_E = {} #Cree un diccionario de probabilidad de exito
    prob_F = {} #Cree un diccionario de probabilidad de fracaso
    peso_E = {}
    peso_F = {}
    i = 0
    #Usar el i de arriba para buscar las entidades
    for rows in Matriz.index: #Para toda fila en los indices de la matriz haga
        
        rows = rows[0] 
    
        inf.update({rows : {}}) #Añada un diccionario con llave la fila en inf
        sup.update({rows : {}}) #Añada un diccionario con llave la fila en sup

        for cols in Matriz.columns: #Para toda columna en los indices de la matriz haga

            cols = cols[0]
            
            c = Matriz.loc[rows,cols].to_numpy()[0][0]

            if c != 0: #Si la posición i,j es distinta de 0 haga
                
                d = coeficientes.loc[rows, :].to_numpy()[0][0]

                if c <= d: #Si la posición i,j es menor o igual a un coeficiente realice
                
                    a = coeficientes.loc[rows,:].to_numpy()[0][0] - Matriz.loc[rows,cols] #Restar el JA-Jij
                    a = a.to_numpy()[0]
                    Matriz2.loc[rows,cols] = a #Guardelo en la posición i,j de la matriz
                    inf[rows].update({cols : a}) #Guardelo en el diccionario

                elif c > d: #Si la posición i,j es mayor a un coeficiente realice
                    
                    a = Matriz.loc[rows,cols] - coeficientes.loc[rows,:].to_numpy()[0][0] #Restar el Jij-JA
                    a = a.to_numpy()[0]
                    Matriz2.loc[rows,cols] = a #Guardelo en la posición i,j de la matriz
                    sup[rows].update({cols : a}) #Guardelo en el diccionario
            
            else: 
                1+1
        
        
        prob_E.update({rows:len(inf[rows])/pesos_fila[i]})
        prob_F.update({rows:len(sup[rows])/pesos_fila[i]})
        peso_E.update({rows:np.sum([inf[rows][i] for i in inf[rows]])})
        peso_F.update({rows:np.sum([sup[rows][i] for i in sup[rows]])})
        
        i += 1

        #Idea, volver a copiar el mismo bucle y solo cambiar el valor por la prob_E/F / pesototal_E/F por el valor que ya tiene la matriz
    
    for rows in Matriz.index: #Para toda fila en los indices de la matriz haga
        
        rows = rows[0]
      
        for cols in Matriz.columns: #Para toda columna en los indices de la matriz haga

            cols = cols[0]
            
            c = Matriz.loc[rows,cols].to_numpy()[0][0]

            if c != 0: #Si la posición i,j es distinta de 0 haga
                
                d = coeficientes.loc[rows, :].to_numpy()[0][0]

                if c <= d: #Si la posición i,j es menor o igual a un coeficiente realice
                
                    a = coeficientes.loc[rows,:].to_numpy()[0][0] - Matriz.loc[rows,cols] #Restar el JA-Jij
                    a = a.to_numpy()[0][0]
                    if peso_E[rows] == 0:
                        Matriz2.loc[rows,cols]=prob_E[rows]
                    else:
                        Matriz2.loc[rows,cols] = a*prob_E[rows]/peso_E[rows] #Guardelo en la posición i,j de la matriz
                        
                elif c > d: #Si la posición i,j es mayor a un coeficiente realice
                    
                    a = Matriz.loc[rows,cols] - coeficientes.loc[rows,:].to_numpy()[0][0] #Restar el Jij-JA
                    a = a.to_numpy()[0][0]
                    if peso_F[rows] == 0:
                        Matriz2.loc[rows,cols]=prob_F[rows]
                    else:
                        Matriz2.loc[rows,cols] = a*prob_F[rows]/peso_F[rows] #Guardelo en la posición i,j de la matriz
    
            
    return([Matriz2, prob_E, prob_F, inf, sup])


# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph
# Creditos a https://www.geeksforgeeks.org/python-program-for-dijkstras-shortest-path-algorithm-greedy-algo-7/
class Graph():
 
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]
    
    
    
    def printSolution(self, dist):
        
        columns = ["Distance from Source"]
        self.vector = []
        
        for node in range(self.V):
            
            self.vector.append([dist[node]])
            
        self.Dataframe = pd.DataFrame(self.vector, columns=columns)
 
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):
 
        # Initialize minimum distance for next node
        min = 1e7
 
        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
 
        return min_index
 
    # Function that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src):
 
        dist = [1e7] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
 
        for cout in range(self.V):
 
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)
 
            # Put the minimum distance vertex in the
            # shortest path tree
            sptSet[u] = True
 
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):
                if (self.graph[u][v] > 0 and
                   sptSet[v] == False and
                   dist[v] > dist[u] + self.graph[u][v]):
                    dist[v] = dist[u] + self.graph[u][v]
 
        self.printSolution(dist)
        
        return(self.Dataframe)
# Driver program
#g = Graph(9)
#g.graph = np.array([[0, 4, 0, 0, 0, 0, 0, 8, 0],
 #          [4, 0, 8, 0, 0, 0, 0, 11, 0],
  #         [0, 8, 0, 7, 0, 4, 0, 0, 2],
   #        [0, 0, 7, 0, 9, 14, 0, 0, 0],
    #       [0, 0, 0, 9, 0, 10, 0, 0, 0],
     #      [0, 0, 4, 14, 10, 0, 2, 0, 0],
      #     [0, 0, 0, 0, 0, 2, 0, 1, 6],
       #    [8, 11, 0, 0, 0, 0, 1, 0, 7],
        #   [0, 0, 2, 0, 0, 0, 6, 7, 0]
         #  ])


 
#print(g.dijkstra(0))


 
# This code is contributed by Divyanshu Mehta
