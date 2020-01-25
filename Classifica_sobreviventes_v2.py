#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 08:53:40 2018

@author: calado

survival 	Survival 	0 = No, 1 = Yes
pclass 	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd
sex      Sex 	
Age     	Age in years 	
sibsp 	# of siblings / spouses aboard the Titanic 	
parch 	# of parents / children aboard the Titanic 	
ticket 	Ticket number 	
fare 	Passenger fare 	
cabin 	Cabin number 	
embarked 	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton
"""
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
plt.close("all")

# Configure visualisations
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6



def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = titanic.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
    
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))
    
#get titanic & test csv files as a DataFrame
train = pd.read_csv("train.csv")
test    = pd.read_csv("test.csv")

full = train.append( test , ignore_index = True )
titanic = full[ :891 ]

del train , test

print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)

#plot_correlation_map( titanic )
plt.savefig('Grafico_Correlacao.png')

# Plot distributions of Age of passangers who survived or did not survive
#plot_distribution( titanic , var = 'Age' , target = 'Survived' , row = 'Sex' )

# Excersise 1
# Plot distributions of Fare of passangers who survived or did not survive
# Plot survival rate by Embarked
#plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )
# Excersise 2
# Plot survival rate by Sex
#plot_categories( titanic , cat = 'Sex' , target = 'Survived' )
# Excersise 3
# Plot survival rate by Pclass
#plot_categories( titanic , cat = 'Pclass' , target = 'Survived' )
# Excersise 4
# Plot survival rate by SibSp
#plot_categories( titanic , cat = 'SibSp' , target = 'Survived' )
# Excersise 5
# Plot survival rate by Parch
#plot_categories( titanic , cat = 'Parch' , target = 'Survived' )
plt.savefig('Parch_Survived_v0.pdf')


# Transform Sex into binary values 0 and 1
sexM = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Male' )
sexF = pd.Series( np.where( full.Sex == 'male' , 0 , 1 ) , name = 'Female' )

# Create a new variable for every unique value of Embarked
embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )
embarked.head()

# Create a new variable for every unique value of Embarked
pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )
pclass.head()

# Create dataset
imputed = pd.DataFrame()
# Fill missing values of Age with the average of Age (mean)
#imputed[ 'Age' ] = full.Age.fillna( full.Age.mean() )
imputed[ 'Age' ] = full.Age.fillna( 0 )
# Fill missing values of Fare with the average of Fare (mean)
imputed[ 'Fare' ] = full.Fare.fillna( 0 )
imputed.head()

child = pd.DataFrame()
child = full[ 'Age' ].map( lambda s : 1 if s <=14 else 0)
child.head()

title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

# a map of more aggregated titles
"""Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }

# we map each title
title[ 'Title' ] = title.Title.map( Title_Dictionary )"""
title = pd.get_dummies( title.Title )

#title = pd.concat( [ title , titles_dummies ] , axis = 1 )
title.head()

surname=pd.DataFrame()
surname[ 'surname' ] = full[ 'Name' ].map( lambda s: s.split(',')[0].strip())
surname = pd.get_dummies(surname.surname)

cabin = pd.DataFrame()
# replacing missing cabins with U (for Uknown)
cabin[ 'Cabin' ] = full.Cabin.fillna( 'U' )
# mapping each Cabin value with the cabin letter
cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )
# dummy encoding ...
cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )
cabin.head()
#cabinclass=pd.DataFrame()
#cabinclass=full.Pclass.fillna(4)
##cabinclass=cabinclass.astype(str)
#cabinclass=cabinclass.replace('1','A00')
#cabinclass=cabinclass.replace('2','B00')
#cabinclass=cabinclass.replace('3','C00')
#cabinclass=cabinclass.replace('4','D00')
#cabin[ 'Cabin' ] = full.Cabin.fillna(cabinclass)
#cabin[ 'Cabin' ] = cabin[ 'Cabin' ].astype(str)
#cabin[ 'Cabin' ] = (cabin[ 'Cabin' ].str[0].apply(ord)-65)*100+pd.to_numeric(cabin[ 'Cabin' ].str[1:4],errors='coerce').fillna(0).astype(np.int64)
#cabin[ 'EmptyCabin' ]=full.Cabin.isnull()
#cabin.head()

# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'

ticket = pd.DataFrame()

# Extracting dummy variables from tickets:
ticket[ 'Ticket' ] = full[ 'Ticket' ].map( cleanTicket )
ticket = pd.get_dummies( ticket.Ticket)
#ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )
#ticket[ 'TicketNumber' ] = pd.to_numeric(full[ 'Ticket' ],errors='coerce').fillna(0).astype(np.int64)

ticket.shape
ticket.head()

family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1

# introducing other features based on the family size
#family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
#family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
#family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
family['Parents'] = full['Parch']

family.head()

# Select which features/variables to include in the dataset from the list below:
# imputed , embarked , pclass , sex , family , cabin , ticket, surname,title

#full_X = pd.concat( [ child, pclass, sexM, sexF, title, embarked ] , axis=1 ) deu 0.78947
full_X = pd.concat( [ child, pclass, sexM, sexF, title, embarked, surname ] , axis=1 )
full_X.head()

# Create all datasets that are necessary to train, validate and test models
train_valid_X = full_X[ 0:891 ]
train_valid_y = titanic.Survived
test_X = full_X[ 891: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = 0.75 )
valid_X=train_valid_X
valid_y = train_valid_y
print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)
#plot_variable_importance(train_X, train_y)

model = RandomForestClassifier(n_estimators=100)
#model = SVC()
#model = GradientBoostingClassifier()
#model = KNeighborsClassifier(n_neighbors = 3)
#model = GaussianNB()
#model = MLPClassifier(learning_rate = 'adaptive', hidden_layer_sizes = 999, max_iter = 99999);
#model = LogisticRegression()

model.fit( train_X , train_y )
print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))
#plot_model_var_imp(model, train_X, train_y)

#rfecv = RFECV(estimator = model, step = 1, cv = StratifiedKFold(2), scoring = 'accuracy')
rfecv = RFECV( estimator = model , step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )
#rfecv.fit( train_X , train_y )

test_Y = model.predict( test_X )
test_Y = np.asarray(test_Y, dtype=int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'titanic_pred.csv' , index = False )