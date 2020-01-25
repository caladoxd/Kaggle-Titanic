# Comando para ignorar mensagens de warnings
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas para lidar com tabelas e matrizes
import numpy as np
import pandas as pd

# Algoritmos de modelagem
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Algoritmos auxiliares para modelagem de dados
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Bibliotecas para visualizacao de dados
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
# Fechar todas janelas abertas
plt.close("all")

# Configuracoes de visualizacao
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6
