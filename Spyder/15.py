#model = RandomForestClassifier(n_estimators=100)
#model = SVC()
#model = GradientBoostingClassifier()
#model = KNeighborsClassifier(n_neighbors = 3)
#model = GaussianNB()
model = MLPClassifier(learning_rate = 'adaptive', hidden_layer_sizes = 200, max_iter = 300);
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
