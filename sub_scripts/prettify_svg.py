from bs4 import BeautifulSoup
import os



base_directory = '/n/pfister_lab2/Lab/wisoo/cubicasa5k/colorful/9769'
base_directory = base_directory + '/'

file_names = os.listdir(base_directory)

img_name = None

for idx, file in enumerate(file_names):
    if ".svg" in file:
        img_name = file


svg = open(base_directory+img_name, 'r').read()
soup = BeautifulSoup(svg, 'lxml')

soup = soup.prettify()

print(soup)


