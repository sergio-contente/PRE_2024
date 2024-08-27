import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_dataframe():
	data_path = '../data/CBIR_15-Scene'
	labels_file_path = os.path.join(data_path, 'Labels.txt')

	with open(labels_file_path, 'r') as file:
			labels_content = file.readlines()
	dataset = []

	for line in labels_content:
			parts = line.strip().split()
			directory = parts[0]
			category = parts[1]
			start_idx = int(parts[2])
			end_idx = int(parts[3])
			
			dir_path = os.path.join(data_path, directory)
			if os.path.exists(dir_path):
					images = os.listdir(dir_path)
					images.sort()
					
					for idx in range(start_idx, end_idx + 1):
						img_name = f"{idx}" + '.jpg'
						if img_name in images:
							img_path = os.path.join(dir_path, img_name)
							dataset.append({'image_path': img_path, 'category': category})
							
	df_dataset = pd.DataFrame(dataset)
	print(df_dataset)

	return df_dataset
