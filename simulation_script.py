
import numpy as np
import csv
import pandas as pd


########################################################### Helper functions################################################################

#In: The original raw data
#Out: The maximum and minimum for every variable
def getMinMaxValues(original_data):
	max_values = []
	min_values = []

	for column in original_data:
		
		max_values.append(original_data[column].max())
		min_values.append(original_data[column].min())

	return max_values,min_values




#In: An integer
#out: the number of decimal places of Integer
def num_after_point(x):
    s = str(x)

    if not '.' in s:
        return 0

    if(s[::-1].find('.') == 1):
    	return 0

    if (s[::-1].find('.') == 5):  ## for mean_gm
      	return 6

    if (s[::-1].find('.') > 6):  #for subtypes
      	return 3


    return s[::-1].find('.')



#In: the simulated data
#Out: the simulated data after it has been clipped and matched for percision

def cleanUP(category,arr_value):

#####Index Code###

#value_iter - Variable
#0 - age_scan
#1 - gender
#2 - mean_gm
#3 - tiv
#4 - ADAS13
#5 - ADNI_MEM
#6 - ADNI_EF
#7 - BNTTOTAL
#8 - CLOCKSCOR
#9 - sub1
#10 -sub2
#11 -sub3
#12 -sub4
#13 -sub5
#14 -sub6
#15 -sub7

	iterr = 0
	max_values,min_values = getMinMaxValues(df_csv)

	#iterate through each subject for one of the 9 variables
	for subID in category:

		value_iter = 0

		#Iterate through each variables within one subject
		for value in subID:

			#clip using max and min
			category[iterr][value_iter] = np.clip(category[iterr][value_iter], min_values[value_iter], max_values[value_iter], out=None)


			#Add threshold for the gender. If greater than 0.5 make 1 and if less than 0.5 make 0. 
			if (value_iter == 1):
				category[iterr][value_iter] = np.where(subID[value_iter]> 0.5,1,0)

			else:
				#match the percision for all variables to the original data
				category[iterr][value_iter] = np.around(value,num_after_point(df_csv.loc[arr_value][value_iter]))

			value_iter += 1	
		iterr += 1

	return category



#In: the original raw data 
# MCI values are changed to pMCI and sMCI
def splitMCI(df_csv):

	mci_iter = 0

	for index, row in df_csv.iterrows():

		if (row["DX"] == "MCI" and row["conv_2_ad"] == 1):
			df_csv.loc[mci_iter,"DX"] = "pMCI"


		elif(row["DX"] == "MCI" and row["conv_2_ad"] == 0):
			df_csv.loc[mci_iter,"DX"] = "sMCI"
			

		mci_iter += 1
		
	return df_csv

########################################################################################################################################

#load data
csv_data = pd.read_csv('adni_vcog_reduced_20180919.csv')
df_csv = pd.DataFrame(csv_data)
df_csv.head()

#change MCI under 'DX' to pMCI or sMCI
df_csv = splitMCI(df_csv.copy())


#Drop subjects who have missing values in any of these variables
df_csv.dropna(subset=['age_scan','gender','mean_gm','tiv', 'ADAS13','ADNI_MEM','ADNI_EF','BNTTOTAL','CLOCKSCOR','sub1','sub2','sub3','sub4','sub5','sub6','sub7','DX','flag_status','conv_2_ad','dataset'],
            inplace=True)

#Create a subset of the variables
df_csv = df_csv[['age_scan','gender','mean_gm','tiv', 'ADAS13','ADNI_MEM','ADNI_EF','BNTTOTAL','CLOCKSCOR','sub1','sub2','sub3','sub4','sub5','sub6','sub7','DX','flag_status','conv_2_ad','dataset']]


mean_dataframe  = pd.DataFrame()
gaussian_dataframe = pd.DataFrame()

covariance_dictionary  = {}
correlation_dictionary  = {}



#################################################################################################################

Clinical = ['CN','pMCI','sMCI','Dementia']
Subclass = ['Negative','Non-HPS+','HPS+']


check = 0

for diagnosis in Clinical:
	for status in Subclass:

		temp_df = pd.DataFrame()
		temp_mean_df = pd.DataFrame()
		temp_covariance = pd.DataFrame()


		#These arrays used to fill the columns in csv file
		conversion_Array = []
		clinical_Array = []
		subclass_Array = []
		dataset_Array = []

		for i,row in df_csv.iterrows():
			DX = row['DX']
			flag_status = row['flag_status']
			conv_2_ad = row['conv_2_ad']
			dataset = row['dataset']


			if(DX == diagnosis and status == flag_status):
					temp_df = temp_df.append(row)

					clinical_Array.append(DX)
					subclass_Array.append(status)
					
					conversion_Array.append(conv_2_ad)
					dataset_Array.append(dataset)


		#Determine the mean	
		temp_mean_df = temp_df.mean()
		temp_mean_df['DX'] = diagnosis
		temp_mean_df['flag_status'] = status
		mean_dataframe = mean_dataframe.append(temp_mean_df, ignore_index = True)


		#####
		category = diagnosis +','+status
		temp_df = temp_df[['age_scan','gender','mean_gm','tiv', 'ADAS13','ADNI_MEM','ADNI_EF','BNTTOTAL','CLOCKSCOR','sub1','sub2','sub3','sub4','sub5','sub6','sub7']]


		#Determine the covariance 
		temp_covariance = np.cov(temp_df.T)
		temp_covariance_df = pd.DataFrame(temp_covariance, columns = ['age_scan','gender','mean_gm','tiv', 'ADAS13','ADNI_MEM','ADNI_EF','BNTTOTAL','CLOCKSCOR','sub1','sub2','sub3','sub4','sub5','sub6','sub7'] )
		#temp_covariance_df.rename({1:'age_scan',2:'gender',3:'mean_gm',4:'tiv',5:'ADAS13',6:'ADNI_MEM',7:'ADNI_EF',8:'BNTTOTAL',9:'CLOCKSCOR',10:'sub1',11:'sub2',12:'sub3',13:'sub4',14:'sub5',15:'sub6',16:'sub7'}, axis='index')
		covariance_dictionary[category] = temp_covariance_df

		
		#Determine the Correlation Matrix
		temp_correlation = np.corrcoef(temp_df.T)
		temp_correlation_df = pd.DataFrame(temp_correlation, columns = ['age_scan','gender','mean_gm','tiv', 'ADAS13','ADNI_MEM','ADNI_EF','BNTTOTAL','CLOCKSCOR','sub1','sub2','sub3','sub4','sub5','sub6','sub7'])
		correlation_dictionary[category] = temp_correlation_df


		#Determine the gaussian distribution
		temp_gaussian = np.random.multivariate_normal(temp_df.mean(),temp_covariance_df,temp_df.shape[0])
		arbitrary_value = 75   # an arbitray subject's values is used to match percision
		temp_gaussian = cleanUP(temp_gaussian,arbitrary_value)
		temp_gaussian_df = pd.DataFrame(temp_gaussian, columns = ['age_scan','gender','mean_gm','tiv', 'ADAS13','ADNI_MEM','ADNI_EF','BNTTOTAL','CLOCKSCOR','sub1','sub2','sub3','sub4','sub5','sub6','sub7'])
		

		temp_gaussian_df.insert(temp_gaussian_df.shape[1],'DX',clinical_Array)
		temp_gaussian_df.insert(temp_gaussian_df.shape[1],'flag_status',subclass_Array)
		temp_gaussian_df.insert(temp_gaussian_df.shape[1],'conv_2_ad',conversion_Array)
		temp_gaussian_df.insert(temp_gaussian_df.shape[1],'dataset',dataset_Array)


		gaussian_dataframe = gaussian_dataframe.append(temp_gaussian_df, ignore_index = True)
		


		

##Create CSV file for simulated data		
gaussian_dataframe.to_csv('simulated_data_v2.csv', sep='\t', encoding='utf-8')


