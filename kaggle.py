import kaggle

#base_url= "https://www.kaggle.com/api/v1"


#kaggle.api.dataset_download_files('Fortune 1000')


from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

api.dataset_download_file('winston56/fortune-500-data-2021',
                          file_name='Fortune_1000.csv',
                          path='./')



