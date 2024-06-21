import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# Trabalho de Introdução a modelagem matemática capítulo 3
# I) Geração e crescimento de redes
# Parte A: Geração
# Modelo de Erdös-Rényi

def modelo_erdos_renyi(qtd_nos, p_aresta):
    # Criando o grafo
    G = nx.Graph()
    
    # Adicionando os nós ao grafo
    G.add_nodes_from(range(qtd_nos))
    
    # Varrendo todos os nós e estabelecendo conexões entre eles usando a probabilidade
    for i in range(qtd_nos):
        for j in range(i+1, qtd_nos):
            # Geramos um valor aleatório entre 0 e 1. Se ele for menor que a probabilidade
            # passada, então cria-se a aresta, caso contrário, não ocorre nada 
            if np.random.rand() < p_aresta:
                G.add_edge(i, j)
    return G

# Modelo de Watts-Strogatz

def modelo_watts_strogatz(qtd_nos, qtd_vizinhos, p_redirecionamento):
    # Criando o grafo
    G = nx.Graph()
    
    # Adicionando os nós ao grafo
    G.add_nodes_from(range(qtd_nos))
    
    # Conectando com os vizinhos mais próximos
    for node in (range(qtd_nos)):
        for i in range(1, qtd_vizinhos//2 + 1):
            vizinho = (node + i) % qtd_nos
            G.add_edge(node, vizinho)
            vizinho = (node - i) % qtd_nos
            G.add_edge(node, vizinho)
            
    # Reordena arestas com a probabilidade
    arestas = list(G.edges())
    for aresta in arestas:
        if np.random.rand() < p_redirecionamento:
            no_1 = aresta[0]
            no_2 = aresta[1]
            G.remove_edge(no_1, no_2)
            
            # ligando ao novo no
            novo_no = random.choice(list(set(range(qtd_nos)) - {no_1} - set(G.neighbors(no_1))))
            G.add_edge(no_1, novo_no)
    
    return G

# Rede aleatória com comunidade

def rede_aleatoria_comunidade(nos_por_comunidade, matriz_de_probabilidade):
    # Criando o grafo
    G = nx.Graph()
    
    # Adicionando nós e os atribuindo a uma comunidade
    comunidade = {}
    contagem_nos = 0
    num_comunidades = len(nos_por_comunidade)
    for indice_comunidade in range(num_comunidades):
        for _ in range(nos_por_comunidade[indice_comunidade]):
            G.add_node(contagem_nos)
            comunidade[contagem_nos] = indice_comunidade
            contagem_nos = contagem_nos + 1
            
    # Conectando nós entre comunidades e dentro das comunidades
    nos = list(G.nodes())
    for i in range(len(nos)):
        for j in range(i + 1, len(nos)):
            comunidade_i = comunidade[nos[i]]
            comunidade_j = comunidade[nos[j]]
            if random.random() < matriz_de_probabilidade[comunidade_i, comunidade_j]:
                G.add_edge(nos[i], nos[j])
    return G, comunidade

# Analisando o modelo de Erdös-Rényi    

G1 = modelo_erdos_renyi(30, 0.5)
nx.draw(G1, with_labels=True, font_weight="bold")
plt.show()

graus_erdos_renyi = [d for n, d in G1.degree()]
plt.hist(graus_erdos_renyi, bins=range(max(graus_erdos_renyi)+2), alpha=0.7, color = "blue", edgecolor="black", align="left")
plt.title('Distribuição de Graus Erdös-Rényi')
plt.xlabel('Grau')
plt.ylabel('Frequência')
plt.show()

grau_conectividade_medio_G1 = np.mean(np.array(list(nx.average_degree_connectivity(G1).values())))
clusterizacao_media_G1 = nx.average_clustering(G1)
media_dos_shortest_paths_G1 = nx.average_shortest_path_length(G1)
print("Grau de conectividade médio Erdös-Rényi: ")
print(grau_conectividade_medio_G1)
print("Clusterização média Erdös-Rényi: ")
print(clusterizacao_media_G1)
print("Média dos shortest path Erdös-Rényi: ")
print(media_dos_shortest_paths_G1)

# Analisando o modelo de Watts-Strogatz

G2= modelo_watts_strogatz(30, 16, 0.5)
nx.draw(G2, with_labels=True, font_weight="bold")
plt.show()

graus_watts_strogatz = [d for n, d in G2.degree()]
plt.hist(graus_watts_strogatz, bins=range(max(graus_watts_strogatz)+2), alpha=0.7, color = "blue", edgecolor="black", align="left")
plt.title('Distribuição de Graus Watts-Strogatz')
plt.xlabel('Grau')
plt.ylabel('Frequência')
plt.show()

grau_conectividade_medio_G2 = np.mean(np.array(list(nx.average_degree_connectivity(G2).values())))
clusterizacao_media_G2 = nx.average_clustering(G2)
media_dos_shortest_paths_G2 = nx.average_shortest_path_length(G2)
print("Grau de conectividade médio Watts-Strogatz: ")
print(grau_conectividade_medio_G2)
print("Clusterização média Watts-Strogatz: ")
print(clusterizacao_media_G2)
print("Média dos shortest path Watts-Strogatz: ")
print(media_dos_shortest_paths_G2)

# Analisando o modelo de Rede aleatória com comunidade

nos_por_comunidade = [5, 10, 15]
matriz_de_probabilidade = np.array([[0.3, 0.5, 0.2],
                        [0.5, 0.25, 0.3],
                        [0.2, 0.3, 0.2]])

G3, comunidade = rede_aleatoria_comunidade(nos_por_comunidade , matriz_de_probabilidade)
pos = nx.spring_layout(G3)
colors = [comunidade[node] for node in G3.nodes()]
plt.figure(figsize=(10, 8))
nx.draw(G3, pos, node_color=colors, with_labels=True, cmap=plt.cm.tab20, node_size=500, edge_color='gray')
plt.show()

graus_rede_aleatoria_comunidade = [d for n, d in G3.degree()]
plt.hist(graus_rede_aleatoria_comunidade, bins=range(max(graus_rede_aleatoria_comunidade)+2), alpha=0.7, color = "blue", edgecolor="black", align="left")
plt.title('Distribuição de Graus Rede Aleatória Com Comunidades')
plt.xlabel('Grau')
plt.ylabel('Frequência')
plt.show()

grau_conectividade_medio_G3 = np.mean(np.array(list(nx.average_degree_connectivity(G3).values())))
clusterizacao_media_G3 = nx.average_clustering(G3)
media_dos_shortest_paths_G3 = nx.average_shortest_path_length(G3)
print("Grau de conectividade médio rede aleatória com comunidade: ")
print(grau_conectividade_medio_G3)
print("Clusterização média rede aleatória com comunidade: ")
print(clusterizacao_media_G3)
print("Média dos shortest path rede aleatória com comunidade: ")
print(media_dos_shortest_paths_G3)

# Parte B: Crescimento
# Modelo de anexação uniforme

def anexacao_uniforme(G, num_novos_nos, num_ligacoes):
    num_nos_existentes = len(G.nodes)

    for i in range(num_novos_nos):
        novo_no = num_nos_existentes + i
        G.add_node(novo_no)
        
        # Selecionar nós existentes para se conectar com o novo nó
        nos_existentes = list(G.nodes)
        nos_existentes.remove(novo_no)
        
        # Selecionar `num_ligacoes` nós de forma uniforme
        nos_para_conectar = random.sample(nos_existentes, num_ligacoes)
        
        # Adicionar arestas entre o novo nó e os nós selecionados
        for no in nos_para_conectar:
            G.add_edge(novo_no, no)
            
# Modelo de Anexação preferencial

def anexacao_preferencial(G, num_novos_nos, num_ligacoes):
    num_nos_existentes = len(G.nodes)

    for i in range(num_novos_nos):
        novo_no = num_nos_existentes + i
        G.add_node(novo_no)
        
        # Obter a lista de nós existentes e suas respectivas quantidades de ligações
        nos_existentes = list(G.nodes)
        nos_existentes.remove(novo_no)
        graus = [G.degree(no) for no in nos_existentes]
        
        # Selecionar nós existentes com probabilidade proporcional ao grau
        nos_para_conectar = random.choices(nos_existentes, weights=graus, k=num_ligacoes)
        
        # Adicionar arestas entre o novo nó e os nós selecionados
        for no in nos_para_conectar:
            G.add_edge(novo_no, no)

# Modelo de Price

def modelo_price(G, num_novos_nos, num_ligacoes, proporcao_preferencial):
    num_nos_existentes = len(G.nodes)

    for i in range(num_novos_nos):
        novo_no = num_nos_existentes + i
        G.add_node(novo_no)
        
        # Calcular o número de ligações preferenciais e uniformes
        num_ligacoes_preferenciais = int(num_ligacoes * proporcao_preferencial)
        num_ligacoes_uniformes = num_ligacoes - num_ligacoes_preferenciais
        
        # Obter a lista de nós existentes e suas respectivas quantidades de ligações
        nos_existentes = list(G.nodes)
        nos_existentes.remove(novo_no)
        
        # Selecionar nós existentes com probabilidade proporcional
        # ao grau (anexação preferencial)
        graus = [G.degree(no) for no in nos_existentes]
        nos_para_conectar_preferencial = random.choices(nos_existentes,
                                                        weights=graus,
                                                        k=num_ligacoes_preferenciais)
        
        # Selecionar nós existentes de forma uniforme (anexação uniforme)
        nos_para_conectar_uniforme = random.sample(nos_existentes, num_ligacoes_uniformes)
        
        # Combinar os nós selecionados
        nos_para_conectar = nos_para_conectar_preferencial + nos_para_conectar_uniforme
        
        # Adicionar arestas entre o novo nó e os nós selecionados
        for no in nos_para_conectar:
            G.add_edge(novo_no, no)
            
# Visualizando o modelo de anexação uniforme

G1 = nx.erdos_renyi_graph(10, 0.25)
nx.draw(G1, with_labels=True)
plt.show()

anexacao_uniforme(G1, 3, 2)
nx.draw(G1, with_labels=True)
plt.show()

# Visualizando o modelo de anexação preferencial

G2 = nx.erdos_renyi_graph(10, 0.25)
nx.draw(G2, with_labels=True)
plt.show()

anexacao_preferencial(G2, 3, 2)
nx.draw(G2, with_labels=True)
plt.show()

# Visualizando o modelo de price

G3 = nx.erdos_renyi_graph(10, 0.25)
nx.draw(G3, with_labels=True)
plt.show()

modelo_price(G3, 3, 2, 0.7)
nx.draw(G3, with_labels=True)
plt.show()

# II) Análise de pontos específicos em redes
# Utilizando pandas para ler as redes do arquivo redes.

dfa = pd.read_excel(r'C:\Users\jguil\Downloads\redes.xlsx', sheet_name='redea')
dfb = pd.read_excel(r'C:\Users\jguil\Downloads\redes.xlsx', sheet_name='redeb')
dfc = pd.read_excel(r'C:\Users\jguil\Downloads\redes.xlsx', sheet_name='redec')
dfd = pd.read_excel(r'C:\Users\jguil\Downloads\redes.xlsx', sheet_name='reded')

Ga = nx.from_pandas_edgelist(dfa, 'source', 'target')
Gb = nx.from_pandas_edgelist(dfb, 'source', 'target')
Gc = nx.from_pandas_edgelist(dfc, 'source', 'target')
Gd = nx.from_pandas_edgelist(dfd, 'source', 'target')

# Medidas
# Grau

lista_nos = [1, 2, 5, 9, 17]

print("GRAU A")
print(dict(Ga.degree([1, 2, 5, 9, 17])))

print("GRAU B")
print(dict(Gb.degree([1, 2, 5, 9, 17])))

print("GRAU C")
print(dict(Gc.degree([1, 2, 5, 9, 17])))

print("GRAU D")
print(dict(Gd.degree([1, 2, 5, 9, 17])))

# Closeness Centrality

print("Closeness Centrality A")
cca = nx.closeness_centrality(Ga)
print([cca[key] for key in lista_nos])

print("Closeness Centrality B")
ccb = nx.closeness_centrality(Gb)
print([ccb[key] for key in lista_nos])

print("Closeness Centrality C")
ccc = nx.closeness_centrality(Gc)
print([ccc[key] for key in lista_nos])

print("Closeness Centrality D")
ccd = nx.closeness_centrality(Gd)
print([ccd[key] for key in lista_nos])

# Betweeness centrality

print("Betweenness Centrality A")
bca = nx.betweenness_centrality(Ga)
print([bca[key] for key in lista_nos])

print("Betweenness Centrality B")
bcb = nx.betweenness_centrality(Gb)
print([bcb[key] for key in lista_nos])

print("Betweenness Centrality C")
bcc = nx.betweenness_centrality(Gc)
print([bcc[key] for key in lista_nos])

print("Betweenness Centrality D")
bcd = nx.betweenness_centrality(Gd)
print([bcd[key] for key in lista_nos])

# Page Rank

print("Page Rank A")
pa = nx.pagerank(Ga)
print([pa[key] for key in lista_nos])

print("Page Rank B")
pb = nx.pagerank(Gb)
print([pb[key] for key in lista_nos])

print("Page Rank C")
pc = nx.pagerank(Gc)
print([pc[key] for key in lista_nos])

print("Page Rank D")
pd = nx.pagerank(Gd)
print([pd[key] for key in lista_nos])


