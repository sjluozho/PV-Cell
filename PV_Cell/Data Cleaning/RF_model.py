from sklearn.ensemble import RandomForestRegressor 

def RFregress(X, y):
    
    modelRF = RandomForestRegressor()
    modelRF.fit(X, y)
    return modelRF