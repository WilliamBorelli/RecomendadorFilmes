# coding: utf-8

# In[16]:

from numpy import *


# In[17]:

# define o número de filmes na 'base de dados'

numero_filmes = 10


# In[18]:

# define o numero de usuarios usados na base de dados

numero_usuarios = 5


# In[19]:

# inicializa algumas notas de filmes aleatoriamente
# uma matrix de 10 x 5

notas = random.randint(11, size = (numero_filmes, numero_usuarios))


# In[20]:

print notas


# In[21]:

# cria uma matrix lógica (matrix que representa se uma nota foi dada ou não)

nota_dada = (notas != 0) * 1


# In[22]:

print nota_dada


# In[23]:

# O que vai ocorrer se não multiplicar por 1

print (notas != 0)


# In[24]:

print (notas != 0) * 1


# In[25]:

# Captura as dimensões de uma matrix usando a propriedade 'shape'

notas.shape


# In[26]:

nota_dada.shape


# In[27]:

# Hora dar algumas notas. Um vetor de 10 x 1 para armazenar todas as notas que serão dadas 

minhas_notas = zeros((numero_filmes, 1))
print minhas_notas


# In[28]:

# Estruturas de dados do Python são iniciadas em 0

print minhas_notas[10] 


# In[29]:

# Dei nota a 3 filmes

minhas_notas[0] = 8
minhas_notas[4] = 7
minhas_notas[7] = 3

print minhas_notas


# In[30]:

# Atualizar notas e nota_dada

notas = append(minhas_notas, notas, axis = 1)
nota_dada = append(((minhas_notas != 0) * 1), nota_dada, axis = 1)


# In[31]:

print notas


# In[32]:

notas.shape


# In[33]:

nota_dada


# In[34]:

print nota_dada

# In[35]:

nota_dada.shape

# In[36]:

#Explicação simples do que significa normalizar um bando de dados

a = [10, 20, 30]
aSoma = sum(a)

# In[37]:

print aSoma

# In[38]:

aMedia = aSoma / 3

# In[39]:

print aMedia


# In[40]:

aMedia = mean(a)
print aMedia


# In[41]:

a = [10 - aMedia, 20 - aMedia, 30 - aMedia]
print a


# In[42]:

print notas

# In[43]:

# uma função que normaliza a base de dados

def normalizar_notas(notas, nota_dada):
    normalizar_notas_filmes = notas.shape[0]
    
    media_notas = zeros(shape = (numero_filmes, 1))
    notas_normalizadas = zeros(shape = notas.shape)
    
    for i in range(numero_filmes): 
        # Pega todos os índices onde existe um 1
        idx = where(nota_dada[i] == 1)[0]
        # Calcular a média de notas dos filmes que apenas tiveram notas dadas pelos usuários
        media_notas[i] = mean(notas[i, idx])
        notas_normalizadas[i, idx] = notas[i, idx] - media_notas[i]
    
    return notas_normalizadas, media_notas


# In[44]:

# Normalizar notas

notas, media_notas = normalizar_notas(notas, nota_dada)


# In[45]:

# Atualiza algumas variáveis chaves

numero_usuarios = notas.shape[1]
numero_caracteristicas = 3


# In[46]:

# Simples explicação do que significa vetorizar uma regressão linear

X = array([[1, 2], [1, 5], [1, 9]])
Theta = array([[0.23], [0.34]])


# In[47]:

print X


# In[48]:

print Theta


# In[49]:

Y = X.dot(Theta)
print Y


# In[50]:

# Inicializar o parâmetro theta (preferencias_usuario), X (caracteristicas_filmes)

caracteristicas_filmes = random.randn( numero_filmes, numero_caracteristicas )
preferencias_usuario = random.randn( numero_usuarios, numero_caracteristicas )
initial_X_and_theta = r_[caracteristicas_filmes.T.flatten(), preferencias_usuario.T.flatten()]


# In[51]:

print caracteristicas_filmes


# In[52]:

print preferencias_usuario


# In[53]:

print inicial_X_e_theta


# In[54]:

inicial_X_e_theta.shape


# In[55]:

caracteristicas_filmes.T.flatten().shape


# In[56]:

preferencias_usuario.T.flatten().shape


# In[57]:

inicial_X_e_theta


# In[58]:

def liberar_parametros(X_e_Theta, numero_usuarios, numero_filmes, numero_caracteristicas):
	# Recuperar as matrizes X e theta de X_e_Theta, baseado em suas dimensões (numero_caracteristicas, numero_filmes, numero_filmes)

	# Conseguir as primeiras 30 (10 * 3) linhas da coluna de um vetor de 48 x 1
	primeiros_30 = X_e_Theta[:numero_filmes * numero_caracteristicas]
	# Remodelar a coluna desse vetor em uma matriz de 10 x 3
	X = primeiros_30.reshape((numero_caracteristicas, numero_filmes)).transpose()
	# Pegar o resto dos 18 números, depois dos primeiros 30
	ultimos_18 = X_e_Theta[numero_filmes * numero_caracteristicas:]
	# Remodelar a coluna desse vetor em uma matriz de 6 x 3
	theta = ultimos_18.reshape(numero_caracteristicas, numero_usuarios ).transpose()
	return X, theta


# In[59]:

def calcular_gradiente(X_e_Theta, notas, nota_dada, numero_usuarios, numero_filmes, numero_caracteristicas, parametro_reg):
	X, theta = liberar_parametros(X_e_Theta, numero_usuarios, numero_filmes, numero_caracteristicas)
	
	# aqui multiplicamos por nota_dada porque porque queremos apenas considerar observações para as quais uma nota foi dada
	diferenca = X.dot( theta.T ) * nota_dada - notas
	gradiente_X = diferenca.dot( theta ) + parametro_reg * X
	gradiente_theta = diferenca.T.dot( X ) + parametro_reg * theta
	
	# colcando os gradientes de volta na coluna do vetor
	return r_[gradiente_X.T.flatten(), gradiente_theta.T.flatten()]


# In[60]:

def calcular_custo(X_e_Theta, notas, nota_dada, numero_usuarios, numero_filmes, numero_caracteristicas, parametro_reg):
	X, theta = liberar_parametros(X_e_Theta, numero_usuarios, numero_filmes, numero_caracteristicas)
	
	# aqui multiplicamos por nota_dada porque porque queremos apenas considerar observações para as quais uma nota foi dada
	custo = sum( (X.dot( theta.T ) * nota_dada - notas) ** 2 ) / 2
	
	regularizacao = (parametro_reg / 2) * (sum( theta**2 ) + sum(X**2))
	return custo + regularizacao


# In[64]:

# importar essas otimizações avançadas (por exemplo a descida gradiente)

from scipy import optimize


# In[65]:

# parâmetro de regularização

parametro_reg = 30


# In[67]:

# executar descida gradiente, econtrar o custo mínimo (csoma dos erros dos quadrados) e otimizar os valores de X (caracteristicas_filmes) e Theta (preferencias_usuario)

custo_minimizado_e_parametros_otimizados = optimize.fmin_cg(calcular_custo, fprime=calcular_gradiente, x0=inicial_X_e_theta, 								args=(notas, notas_dadas, numero_usuarios, numero_filmes, numero_caracteristicas, parametro_reg), 								maxiter=100, disp=True, full_output=True ) 


# In[ ]:

custo, caracteristicas_otimizadas_e_preferencias_usuario = custo_minimizado_e_parametros_otimizados[1], custo_minimizado_e_parametros_otimizados[0]


# In[ ]:

# liberar novamente

caracteristicas_filmes, preferencias_usuario = liberar_parametros(caracteristicas_otimizadas_e_preferencias_usuario, numero_usuarios, numero_filmes, numero_caracteristicas)


# In[ ]:

print caracteristicas_filmes


# In[ ]:

print preferencias_usuario


# In[ ]:

# Vamos fazer algumas previsões (recomendações de filmes)

todas_previsoes = caracteristicas_filmes.dot( preferencias_usuario.T )


# In[ ]:

print todas_previsoes


# In[ ]:

# adicionando novamente a coluna do vetor media_notas para minhas próprias previsões

previsoes_para_william = todas_previsoes[:, 0:1] + media_notas


# In[ ]:

print previsoes_para_william


# In[ ]:

print minhas_notas


# In[ ]: