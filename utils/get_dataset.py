import os
from shutil import copy, move

images_path = 'C:\\Users\\HashRoot\\Downloads\\idd-fine\\anue\\leftImg8bit\\train\\'
json_anns_path = 'C:\\Users\\HashRoot\\Downloads\\idd-fine\\anue\\gtFine\\train\\'

images_save_path = 'C:\\Users\\HashRoot\\Downloads\\public-code-master\\preperation\\IDD\\Train\\Images\\'
anns_save_path = 'C:\\Users\\HashRoot\\Downloads\\public-code-master\\preperation\\IDD\\Train\\JSON_Annotations\\'

images_folders = os.listdir(images_path)

total_files_saved = 0

for folder in images_folders:
    images_folder_path = images_path + folder + '\\'
    anns_folder_path = json_anns_path + folder + '\\'
    for ele in os.listdir(images_folder_path):
        # print(ele)
        image_file = ele
        ann_file = ele.split('_')[0] + '_gtFine_polygons.json'

        image_path = images_folder_path + image_file
        ann_path = anns_folder_path + ann_file

        if not os.path.exists(image_path) and os.path.exists(ann_path):
            print(image_path, ann_path)
            print('File Not Found')
        move(src=image_path, dst=images_save_path+image_file)
        move(src=ann_path, dst=anns_save_path + ann_file)

        # print('Saved File')

        total_files_saved += 1
        print(total_files_saved)
print('Total Files Saved : ', total_files_saved)
    
