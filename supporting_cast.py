import pandas as pd

##  data pre pre pre processing
airlineData = pd.read_csv(r"C:\Users\Matt\Documents\Pytorch_ML\Generative\Airline DS from scratch\data\US Airline Flight Routes and Fares.csv")

#removing columns I don't care about
toberemoved = ["tbl", "Year", "city1", "city2", "citymarketid_1", "citymarketid_2", "Geocoded_City1", "Geocoded_City2", "tbl1apk"]
airlineData.drop(toberemoved, axis = 1, inplace = True)     #inplace removal of toberemoved

#save to two new DF everything with WN as the carrier_lg or carrier_low
#aiq - "airline in question" 

AIQ = "NW"                  #temporarily setting airline to southwest
section = 'carrier_low'     #temporarily setting the section of interest to be the lowest (cheapest carrier). This will change in section A3


airlineData = airlineData.dropna(axis = 0, subset = section)                  #drop NaN values to make the next line valid
AIQ_section = airlineData[airlineData[section].str.contains(AIQ)]             #temporarily, I'll shrink the dataset to just be the subset with the airline in question
AIQ_section.reset_index()

##  data pre pre processing 
#(A1 part i)
columnsOfInterest = ["nsmiles", "passengers", "fare", "large_ms", "fare_lg", "lf_ms" ,"fare_low"]
temporarilyDropTextColumns = ["airport_1", "airport_2", "carrier_lg", "carrier_low"] 

airlineSimplified = AIQ_section.copy(columnsOfInterest)                             #Copy function allows us to copy the columns I care about for the covariance matrix
airlineSimplified.drop(temporarilyDropTextColumns, axis = 1, inplace = True)        #inplace = True dropping some remaining Text columns I figure will be useful later but aren't right nows (and will cause error with covariance matrix obviously)
covMatrix = airlineSimplified.copy()                                                #(B1) save this as it's the starting point for B1 and onwards

covMatrix["revenue"] = covMatrix["passengers"].mul(covMatrix["fare"])
covMatrix["rangePrices"] = covMatrix["fare_lg"] - covMatrix["fare_low"]

covarianceMatrix = covMatrix.cov()
correlationMatrix = covMatrix.corr()

#covarianceMatrix.to_csv(r"C:\Users\Matt\Documents\Pytorch_ML\Generative\Airline DS from scratch\data\covMatrix.csv")
#correlationMatrix.to_csv(r"C:\Users\Matt\Documents\Pytorch_ML\Generative\Airline DS from scratch\data\corrMatrix.csv")

'''
on the fare_low, the highest + to lowest - impact items were as follows
    fare, 0.77935
    fare_lg, 0.73089
    ns_miles. 0.32674
    lf_ms, 0.2347
    quarter, -0.04412
    passengers, -0.12588
    large_ms, -0.13553

Therefore, as predictors, we will use ns_miles, lf_ms, passengers, and large_ms as the appropriate predictors
Conveniently, the number of miles of a flight can be calculated, and the passengers is a shrimple input.
The lf_ms and large_ms can be predicted for a given date using historical data, which will probably be a seperate shrimple regression prediction.     
'''
#for now, I'm going to cut out just the 3 ns_miles, lf_ms, passengers, and large_ms, getting rid of any other columns

columnsOfInterest_afterDataPostProcessing = ["ns_miles", "passngers", "lf_ms", "large_ms", "fare", "fare_low", "fare_ms"]
airlineSimplified = airlineSimplified.copy(columnsOfInterest_afterDataPostProcessing)   
#airlineSimplified.to_csv(r"C:\Users\Matt\Documents\Pytorch_ML\Generative\Airline DS from scratch\data\airlineSimplifiedOutput.csv")

#On the other two fare prices, fare and fare_lg, the numbers vary (albiet not by much), so look at those individually when I get there

